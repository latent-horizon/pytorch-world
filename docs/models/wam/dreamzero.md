# DreamZero

> Ye et al., *World Action Models are Zero-shot Policies*, ICLR 2026.
> [Paper](https://arxiv.org/abs/2602.15922) · [Notebook](../../examples/action/dreamzero.ipynb)

DreamZero turns an image-to-video diffusion model into a policy. One causal diffusion transformer denoises the next **chunk** of video latents and the **action chunk** that produces them, with a single flow-matching loss. Chunk `k` holds `K` latent frames `z^k`, the `H` actions `a^k` executed over them and the proprioceptive state `q_k` at its start. Video and actions are interpolated from Gaussian noise with a shared time `t_k`, and the network regresses the joint velocity:

$$
\mathcal{L} = \frac{1}{M}\sum_{k=1}^{M} \big\lVert u_\theta\big([z^k_{t_k}, a^k_{t_k}];\ C_k, q_k, t_k\big) - \big([z^k, a^k] - [\epsilon_z, \epsilon_a]\big) \big\rVert^2
$$

The context `C_k` holds the clean earlier chunks. Training is **chunk-wise teacher forcing**: all `M` chunks of a clip are noised, each with its own time, and appended after the clean chunks in one sequence. A block-causal mask lets noisy chunk `k` attend to the anchor frame (and the goal frame, if any), the clean chunks before `k` and itself. One forward pass trains every autoregressive step.

At inference, the next chunk is sampled by integrating the flow from noise. After the robot executes the chunk, the real observations are encoded and **replace the predicted video in the context**, so imagination errors never compound across chunks.

```python
from torch_world.models import DreamZero
from torch_world.nn import VideoVAE
from torch_world.planning import DreamZeroPolicy, WAMPlanner

vae = VideoVAE.from_pretrained()                     # Wan 2.1: 16 channels, 8x spatial, 4x temporal
model = DreamZero(
    16, (12, 12), action_dim=4, patch_size=2,        # latents of 96 x 96 frames
    chunk_frames=2, chunk_actions=8, num_chunks=4,   # a chunk: 2 latent frames = 8 video frames, 8 actions
    dim=384, depth=8, num_heads=6, goal_conditioned=True, autoencoder=vae,
)
# latents: (B, 1 + 4 * 2, 16, 12, 12) from vae.encode of 33 frames; actions: (B, 32, 4); goal: (B, 16, 12, 12)
losses = model(latents, actions, goal=goal)
losses["loss"].backward()

model.eval()
future, chunk = model.imagine(z, goal, past_actions=past)   # next 2 latent frames and 8 actions
video = model.decode(torch.cat([z, future], 1))              # back to pixels
policy = DreamZeroPolicy(model, goal)                        # closed loop: policy(observation) -> action
ranked = DreamZeroPolicy(model, goal, plan=WAMPlanner(num_samples=8))
```

`imagine` returns latents `(B, K, C, h, w)` and physical actions `(B, H, A)`. `sample_actions` returns only the actions. `predict_future(z, actions, goal)` denoises video given clean actions, differentiably in the actions. `rollout` chains it over several chunks, so `CEM` also works. The context `z` is an anchor frame followed by whole chunks, with `past_actions` the actions executed over them. Context longer than `num_chunks - 1` chunks is cut, and the frame before the window becomes the anchor, so inference always matches a training position.

## Training at scale

`torch_world.benchmarks.dreamzero` runs the whole pipeline with one process per GPU, on one node or many:

| command | splits across ranks | writes |
|---|---|---|
| `prepare` | episodes: each rank collects play data, encodes clips with the Wan VAE and writes its own shards | `data/shard-*.pt`, `data/config.json` |
| `train` | clips, through a `DistributedSampler`; DDP replicates the model, `--strategy fsdp` shards each transformer block, the parameters, gradients and optimizer state | `log.jsonl`, `checkpoint.pt` (resumable), `model.pt` |
| `evaluate` | goal-reaching episodes, gathered on rank 0 with Wilson intervals | `evaluation-<task>.json` |
| `diagnose` | — (one process) | `diagnose.json`: open-loop errors on held-out shards |

`prepare --data reach` writes noisy scripted reaching demonstrations instead of play data. Each clip's last frame is its goal, so every goal comes with actions that move toward it. `train` keeps an exponential moving average of the weights (`--ema 0.999`) and saves it as `model.pt`, the weights `evaluate` and `diagnose` load; the raw weights stay in `checkpoint.pt`.

`diagnose` tells a weak video model from a weak action branch. From an anchor frame and one real chunk, it compares the policy's actions with `inverse_dynamics`, which denoises actions with the *true* next-chunk video given clean. A large gap means imagined video is the bottleneck. Training the same network with `--video-weight 0` gives the matching policy without video prediction, which measures what predicting video adds.

```sh
torchrun --nproc-per-node 8 -m torch_world.benchmarks.dreamzero prepare --output runs/dz/data --episodes 8192
torchrun --nproc-per-node 8 -m torch_world.benchmarks.dreamzero train --data runs/dz/data --output runs/dz/train \
    --strategy fsdp --dim 1024 --depth 24 --num-heads 16 --batch-size 16 --accumulate 4 --steps 50000
torchrun --nproc-per-node 8 -m torch_world.benchmarks.dreamzero evaluate --run runs/dz/train --episodes 1024
```

Across nodes, launch one `torchrun` per node with a shared rendezvous. Under Slurm:

```sh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
HEAD=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
srun python -m torch.distributed.run --nnodes $SLURM_NNODES --nproc-per-node 8 \
    --rdzv-id $SLURM_JOB_ID --rdzv-backend c10d --rdzv-endpoint $HEAD:29500 \
    -m torch_world.benchmarks.dreamzero train --data runs/dz/data --output runs/dz/train --strategy fsdp
```

What keeps it efficient as clips get longer and larger:

- **Attention.** `attention="flex"` (the script's default) runs the block-causal mask as a compiled FlexAttention `BlockMask`, which skips the masked half of the attention matrix instead of materializing it. Masks are built once per sequence layout. FlexAttention trains on CUDA only; `attention="sdpa"` works everywhere.
- **Inference.** `context_cache` computes the keys and values of the clean context once per chunk. Each denoising step then runs only the new chunk's tokens, so sampling costs one context pass plus `flow_steps` passes over a single chunk.
- **Precision.** DDP trains under bf16 autocast. FSDP2 keeps fp32 master weights and computes in bf16, reducing gradients in fp32.
- **Effective batch.** The global batch is `batch_size * accumulate * world_size`. Gradient synchronization is skipped on all but the last micro-batch.
- **Checkpoints.** Checkpoints hold the full, unsharded state, so a run can resume on a different number of GPUs. The frozen VAE is never part of the trained model or its checkpoints.

`prepare` and `evaluate` use `RobotCube`. For other data, write shards with `latents` `(N, 1 + num_chunks * chunk_frames, C, h, w)`, `actions` `(N, num_chunks * chunk_actions, A)`, optionally `goal` `(N, C, h, w)` and `states` `(N, num_chunks, S)`, plus a `config.json` with `latent_channels`, `latent_size`, `action_dim`, `chunk_frames`, `chunk_actions` and `num_chunks`.

## DreamZero-Flash

`flash_alpha=7` trains as DreamZero-Flash does. It draws video times from `1 - Beta(7, 1)` (mean 0.125, mostly noise) and action times from `U(0, 1)`. The model learns to produce clean actions while the video is still noisy, which is what one-step sampling needs. The paper fine-tunes with Flash after standard training: reload the weights into a model with `flash_alpha` set, then train on.

## Implemented scope

Implemented as in the paper:

- joint video-action flow matching with a shared time per chunk and independent times across chunks;
- chunk-wise teacher forcing with clean previous chunks, both video and actions, as context;
- `K = 2` latent frames per chunk by default;
- state, action encoders and an action decoder around a video DiT;
- frozen Wan 2.1 VAE latents (`VideoVAE`);
- autoregressive inference with real-observation replacement (`DreamZeroPolicy`);
- Flash time sampling.

Differences from the paper:

- **No pretrained video backbone.** The DiT is trained from scratch at small scale, not fine-tuned from Wan2.1-I2V-14B, so this reproduces the recipe and not the zero-shot results.
- **Goal image instead of language.** An optional goal latent frame replaces the text prompt; there is no text or CLIP image encoder.
- **Attention within a chunk.** Tokens attend bidirectionally. The paper describes chunk-level causality only.
- **Positions.** Video tokens use fixed 3D sin-cos positions and action tokens 1D positions. Wan's 3D RoPE is not used.
- **Options beyond the paper.** `time_schedule="logit_normal"` samples flow times from `sigmoid(N(0, 1))`, as Stable Diffusion 3 does, instead of the paper's uniform times. `video_weight` scales the video loss; 0 is the no-video ablation.
- **Unspecified details.** The paper does not specify the action and state encoders, decoder or loss weight. Here they are two-layer MLPs, one token per action, and the loss is the mean squared error on video plus `action_weight` times the error on actions. The paper does not give a timestep shift either, so times are uniform.
- **Inference.**
  - Euler integration replaces Flow UniPC.
  - There is no DiT caching and no Savitzky-Golay action smoothing.
  - The closed-loop policy re-encodes the real frames of the context window with the VAE at each chunk. The transformer side keeps a KV cache of the context.
- **Single view.** Multi-view inputs (concatenated into one frame in the paper) are not set up.

`VideoVAE` needs `pip install torch-world[wam]` and downloads the Wan 2.1 VAE from the Hugging Face Hub. `ConvVideoAE` is a small trainable stand-in with the same interface and no download.

## API

::: torch_world.models.DreamZero

::: torch_world.planning.DreamZeroPolicy
