# LeWAM

> Fu et al., *LeWAM: End-to-End World and Action Modeling with JEPAs*, 2026.
> [Paper](https://arxiv.org/abs/2610.12818) · [Reference code](https://github.com/MinghaoFu/lewam) · [Action notebook](../../examples/action/lewam.ipynb)

LeWAM jointly learns a latent world model and a policy that generates action chunks through flow matching. A mixture of transformers uses separate state/action projections and feed-forward networks, with shared attention. Future-state queries attend to observation history and causally available clean actions. Noisy-action queries attend to history, a goal when configured, and their own causal prefix. History never attends to either prediction branch. This prevents the policy from reading its clean training targets, including indirectly through history.

The action branch learns the velocity of a linear interpolation from Gaussian noise to demonstrated actions. At inference, Euler integration generates a chunk. The dynamics branch predicts future embeddings with MSE. The jointly trained encoder receives gradients through both inputs and targets; SIGReg regularizes its visual embeddings against collapse.

```python
from torch_world.models import LeWAM
from torch_world.nn import CLSEncoder
from torch_world.planning import LeWAMPlanner

model = LeWAM(
    CLSEncoder(img_size=32, patch_size=8, embed_dim=64, depth=2, num_heads=4),
    action_dim=2, embed_dim=64, history=3, chunk_size=8,
    dim=64, depth=2, num_heads=4, mlp_dim=256,
)
# pixels: (B, 11, 3, 32, 32); actions: (B, 10, 2), in environment units.
# goal_pixels: (B, 3, 32, 32), sampled from the SAME episode/task.
losses = model(pixels, actions, goal_pixels=goal_pixels)
losses["loss"].backward()

model.eval()
z = model.encode(pixels[:, :3])
goal = model.encode(goal_pixels[:, None])[:, 0]
# Direct policy: no model-based selection.
actions = LeWAMPlanner()(model, z, goal)
# Optional model-based ranking and trust-region gradient refinement.
actions = LeWAMPlanner(num_samples=16, refinement_steps=3)(model, z, goal)
# Execute the first action, observe again, then replan.
```

`predict_future(z, actions)` returns `num_states` latents spaced by `frameskip` environment steps. The legacy defaults predict every action step. `predict` and the generic `WorldModel.rollout` require `frameskip=1`; spaced models must use `predict_future` so a five-step prediction is never mistaken for a one-step transition. `sample_actions` and the planner return physical-unit actions bounded by ±`action_scale`. The default scale is one; use per-coordinate scales for mixed Cartesian/gripper controls. Supply `action_mean` and `action_std`, fitted on training data only, for reference z-score normalization. Physical bounds apply after denormalization; a valid action several standard deviations from the mean is not truncated to one standard deviation.

## Implemented scope

This is a single-view implementation of **regression dynamics + flow-matching actions**, audited against [reference commit `f57659f`](https://github.com/MinghaoFu/lewam/tree/f57659fb26858082975794d37a8f3d380ddc1fa7). It supports independently configured dynamics spacing (`frameskip`), predicted states (`num_states`) and action chunk length (`chunk_size`), padded-history attention masks, terminal image goals, task-only control and optional normalized horizon conditioning. It is not an upstream checkpoint loader or a reproduction of the paper's scores. Multi-view layouts, flow-based state prediction and the full reference evaluation suite remain outside this implementation. Sampled-future goals and single-view MSE multi-step gradient planning are implemented.

The default encoder projector remains the legacy MLP, preserving existing checkpoint keys. `LeWAMEncoder` supplies the reference ResNet18/GN16 trunk, 32 spatial-softmax keypoints, and random-training/center-evaluation crops. Use `feature_dim=64, projector_norm="batch"` to add the reference BatchNorm projector. This implementation uses PyTorch alone; torchvision is needed only for the optional upstream parity test. SIGReg excludes appended measured-state coordinates.

`LeWAMPlanner` ranks a single chunk toward the goal and can refine it with a squared-distance penalty to its learned proposal. It keeps the best iterate, including the unrefined proposal. For sampled-goal checkpoints, `LeWAMHorizonPlanner` instead alternates policy proposals and predicted states over multiple dynamics steps and refines every proposal with Adam. Neither planner certifies geometric feasibility.

For inference diagnostics, `LeWAMPlanner(flow_steps=32)` changes only the Euler
integration budget. `LeWAMPlanner(noise_scale=0)` starts the flow at zero instead
of a Gaussian draw, giving a deterministic centre-initialized proposal. This is
not the distribution's mean action and is not guaranteed to improve control.
Defaults retain ordinary Gaussian sampling and the model's integration budget.

## Matched upstream/local study

The current notebook compares our implementation with the actual upstream ResNet encoder, transformer, training loss and SIGReg loaded from an explicitly supplied checkout. `lewam_reference.py` loads only the named definitions; no source is downloaded or executed during ordinary library imports. Both variants use the same owned-arm data sampler, initial active weights, training-only action statistics, optimizer schedule and tested planning interface. This is a comparison of the neural/training implementations, not an independent execution of the entire upstream CLI.

Each of 50 epochs visits all 12,288 eligible starts once: batch 128, 96 updates per epoch, 4,800 updates per model. Training starts from scratch with five warmup epochs. Full-loss and encoder-gradient tests couple random draws in float32; the training implementations use their own stochastic paths, so paired seeds do not imply identical updates.

The Spark run uses bfloat16 autocast, an upstream-supported option, with a shared float32 latent boundary. Upstream's default precision is float32. The boundary avoids a PyTorch 2.14 indexed-write error when bfloat16 encoder outputs are combined with float32 positional embeddings; it retains encoder gradients and is applied to both variants. Float32 preflight timing projected several hours on the shared GPU. Initial failed/timing preflights are retained separately and are not reported as completed experiments.

```sh
python -m torch_world.benchmarks.lewam_study --prepare-data --output runs/lewam-matched/data
MUJOCO_GL=egl python -m torch_world.benchmarks.lewam_matched \
  --data runs/lewam-matched/data/episodes.pt --reference "$LEWAM_REFERENCE" \
  --output runs/lewam-matched/training --epochs 50 --batch-size 128 --precision bf16
MUJOCO_GL=egl python -m torch_world.benchmarks.lewam_study \
  --training runs/lewam-matched/training --reference "$LEWAM_REFERENCE" \
  --output runs/lewam-matched/evaluation --cases 50
```

Evaluation uses 50 cases for each of seeds 42, 0 and 1, separately on training trajectories and fresh task trajectories. Initial pose successes, conditional first hits, native first hits, final settling and a final ten-control hold are reported independently. The owned-arm pose tolerance is not the identical DMControl metric. One shared training initialization does not measure variation across trained model seeds. Every controller within a task group shares its starts and goals; expert/zero baselines are reused across models rather than counted as new trials.

`lewam_prediction_audit.py` compares fixed candidate controls at 5, 10 and 25 steps. Ranking actual future-image embeddings tests the goal-distance representation; ranking imagined embeddings also tests the learned dynamics. It reports pairwise agreement with physical goal distance, candidate-selection regret, latent MSE and actual latent variance. Ties are excluded from agreement, and no diagnostic selects checkpoints.

`--prepare-diverse` creates a separate dataset with ordinary approaches, reversed approaches, corrections from either side and braking from nonzero velocity. All 128 training and 16 validation demonstrations in the collected dataset held native success over the final ten recorded states. The baseline matched models do not use this dataset; a subsequent matched run can test coverage without changing physics or tolerances.

The matched Spark study has completed training and evaluation. The notebook displays the complete trial tables, prediction diagnostics and first unseen rollout from `examples/action/reports/lewam-matched.json`.


The executed notebook reports 147/150 unseen pose reaches for both direct policies, but only 84/150 local and 89/150 upstream final settled successes. Planning settles on 8/150 and 9/150 respectively. Similar failures support investigating coverage and planning, rather than claiming paper-level reproduction or hardware readiness.

## Owned robotic-arm environment

`torch_world.envs.ObstacleArm` is the library-owned `ObstacleArm-v0` used by the study. It accepts normalized shoulder/elbow torques and returns batched RGB observations (or state vectors with `observation="state"`). Native success requires both pose accuracy and settling. Its optional render-only goal marker does not affect dynamics; `set_visual_goal` keeps the collection target visible while scoring sampled future poses.

Install `torch-world[lewam,notebooks]` for the recorded MuJoCo version and notebook dependencies. Upstream comparisons additionally require torchvision and an explicit checkout of the pinned reference commit. Ordinary library imports never download or execute upstream source. The archived reports retain their original source hashes; this PR packages the same arm scene and dynamics independently of the broader environment catalog used during the run.

## API

::: torch_world.models.LeWAM

::: torch_world.models.LeWAMEncoder
