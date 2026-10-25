# VLA-JEPA

> Sun et al., *VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model*, 2026. [Paper](https://arxiv.org/abs/2602.10098) · [Code](https://github.com/ginwind/VLA-JEPA) · [Notebook](../../examples/action/vlajepa.ipynb)

VLA-JEPA trains a vision-language-action policy together with a JEPA world model, which lets it learn dynamics from video without labels.

- A causal **VLM** reads the current image and the instruction, followed by two kinds of learnable query tokens. There is one *latent-action* token per transition, repeated $K$ times, and then one *embodied-action* token, repeated 32 times.
- A frozen **target encoder** (V-JEPA 2) embeds every frame of the clip into world states $s_0, \dots, s_{T-1}$. Future frames serve only as regression targets and never reach the VLM, so the policy cannot read the future.
- An action-conditioned **predictor** attends causally across frames and bidirectionally within a frame. At each step it reads the true state with that step's latent-action tokens interleaved, and predicts the next state (teacher forcing).
- A flow-matching **action expert** (a DiT) denoises the next action chunk, conditioned on the embodied-action tokens.

Gradients from the world-model loss reach the VLM only through the latent-action tokens. So the VLM must encode in them what the image and the instruction imply about the future, and the embodied-action tokens, which attend to them, build on that. Human videos without actions train the world-model term alone.

$$
a_t = (1 - t)\,\epsilon + t\,a, \qquad
\mathcal{L} = \big\lVert v_\theta(a_t, t \mid e) - (a - \epsilon) \big\rVert^2
+ \beta \sum_{k=1}^{T-1} \big\lvert P_\phi(s_{<k}, \ell_{<k})_k - s_k \big\rvert
$$

Here $e$ are the embodied-action tokens and $\ell$ the latent-action tokens. At inference only the VLM and the expert run: `sample_actions` integrates the flow from noise in 4 Euler steps.

## Usage

```python
from torch_world.models import VLAJEPA
from torch_world.nn import VideoTokens, VisionTransformer, WordTokenizer

tokenizer = WordTokenizer(instructions)
model = VLAJEPA(
    VisionTransformer(img_size=64, patch_size=8, embed_dim=192, depth=4, num_heads=4),  # trainable vision tower
    VideoTokens.from_hub("vjepa2_1_vit_base_384", size=64),                              # frozen target encoder
    len(tokenizer), action_dim=2, vision_dim=192, target_dim=768, num_patches=16, frames=8, chunk_size=7,
)

ids, mask = tokenizer(texts)
out = model(pixels, ids, mask, actions)   # (B, 8, C, H, W), (B, L), (B, L), (B, 7, A)
# out: {"loss", "action_loss", "wm_loss"}; without actions (video only): {"loss", "wm_loss"}
out = model(pixels[:, :1], ids, mask, actions, latents=z)  # with world states precomputed by model.encode

chunk = model.sample_actions(image, ids, mask)              # (B, 7, A)
chunk, futures = model.imagine(image, ids, mask)            # futures: (B, 8, P, D), the states the VLM expects
```

Run a policy closed loop with `torch_world.planning.RecedingHorizon`: `RecedingHorizon(lambda obs: model.sample_actions(obs, ids, mask), action_dim, execute=4)`.

**Base class.** `VLAJEPA` is a [`WorldActionModel`](../../api/models.md#torch_world.models.WorldActionModel): one network outputs actions and predicts their consequences. Its world model, though, is conditioned on the VLM's latent-action tokens, not on robot actions. So `predict`, `rollout` and `predict_future` take latent-action tokens `(B, T, K, dim)` in place of actions, and action-space planners such as `CEM` do not apply. The policy is the action expert. The world model shapes the VLM during training, and `imagine` shows what it expects under an instruction.

## Choices and differences from the paper

The released code (`starVLA/model/framework/VLA_JEPA.py`) settles several details the paper leaves open. This implementation follows the code on these points:

- The world-model loss is L1, weighted by $\beta = 0.1$. The paper gives neither the distance nor $\beta$.
- The VLM prompt is: image, then *"Your task is {instruction}. Infer the temporal dynamics from frames {latent-action tokens} and produce the corresponding policy actions {embodied-action tokens}."* Each latent-action token is one vocabulary id repeated $K$ times, and the embodied-action token is one id repeated 32 times. Because the VLM is causal, the latent-action tokens cannot see the embodied ones. So video clips, whose prompt omits the embodied tokens, train exactly the same latent tokens. $K$ defaults to 3, the paper's $24/T$; the released config uses 8.
- The predictor follows V-JEPA 2-AC. Per step, it takes $K$ latent-action tokens followed by the state's patch tokens, with a block-causal mask, and it reads the true states (teacher forcing).
- The flow-matching expert follows GR00T N1.5's head. Times are drawn as $1 - \mathrm{Beta}(1.5, 1)$, which concentrates them near noise. Each sample is repeated with fresh noise `flow_repeats = 8` times (`repeated_diffusion_steps`). Inference uses 4 Euler steps.

Some deviations are needed for a self-contained, small model:

- **VLM.** A small causal transformer stands in for Qwen3-VL-2B. It reads the patch tokens of a trainable vision tower, word embeddings (`WordTokenizer`), and the query tokens, with 1-D sin-cos positions. It is trained from scratch, not from a pretrained VLM.
- **Action expert.** GR00T's DiT interleaves cross-attention to the VLM tokens with self-attention, and adds learnable "future" register tokens. Here a single adaLN-zero transformer (`Transformer(conditional=True)`) attends jointly over the projected embodied-action tokens, an optional state token, and the noisy action tokens. The flow time enters through adaLN-zero, as in GR00T.
- **Target encoder.** In the code, V-JEPA 2 encodes clips with tubelet 2, so 8 frames give 4 world states. Here the target encoder embeds each frame separately: with `VideoTokens`, each frame is repeated into a still tubelet. The predictor uses learned patch positions and sin-cos step positions, not 3-D RoPE.
- **Training.** One learning rate for every trainable part. The paper uses 1e-5 for the VLM and world model and 1e-4 for the action head, values meant for fine-tuning pretrained weights. The two-stage recipe (pretraining on SSv2 and DROID, then fine-tuning) becomes a single run on one environment.

## Notebook result

The [notebook](../../examples/action/vlajepa.ipynb) runs on Two-Rooms with eight templated instructions, such as "go to the top left of the right room". The training data is random play, and each clip's instruction is a hindsight label: the region the agent reached 7 to 40 steps later. Each model has 12.2M trainable parameters and trains for 3,000 steps of 128 clips. The full run takes 25 minutes on one GB10. Over 256 episodes of 60 steps, an episode succeeds if the agent enters the named region:

| Policy | Same room (112) | Other room (144) | All (256) |
|---|---|---|---|
| VLA-JEPA | 96% [90%, 98%] | 9% [5%, 15%] | 47% [41%, 53%] |
| without the world-model loss | 98% [94%, 100%] | 10% [6%, 16%] | 49% [43%, 55%] |
| VLA-JEPA, wrong instruction | 39% [31%, 49%] | 8% [5%, 14%] | 22% [17%, 27%] |
| random actions | 62% [52%, 70%] | 10% [6%, 16%] | 32% [27%, 38%] |

Within a room, the policy follows the instruction. Given the wrong one, the same policy falls well below random actions. Reaching the other room through the door fails: random play rarely crosses the door, so hindsight labels for those regions are scarce. In this small setting the world-model term makes no measurable difference, and the ablation scores the same within the intervals. The paper's gains come from co-training on large human-video and robot datasets.

## Reference

Arguments and methods: [`VLAJEPA`](../../api/models.md#torch_world.models.VLAJEPA) in the API reference.
