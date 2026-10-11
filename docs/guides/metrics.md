# Metrics

`torch_world.metrics` implements the metrics the JEPA papers report. Every function takes plain tensors (features, predictions, similarities), so it works with any model.

| question | metric | reported by |
|---|---|---|
| Do frozen features encode a quantity? | `linear_probe` (accuracy for classes, R² for regression) | I-JEPA, LeJEPA, LeWM (physical probing) |
| Same, with no training at all | `knn_accuracy` (weighted k-NN, cosine) | DINO, LeJEPA |
| Same, from all tokens of a video or image | `attentive_probe` | V-JEPA, V-JEPA 2 |
| Has the representation collapsed? | `rankme` (effective rank, no labels) | RankMe; collapse checks in SSL |
| Will linear probes work, without labels? | `lidar` (effective rank of the LDA matrix over views) | LiDAR, designed for JEPAs |
| How fast do imagined latents drift? | `rollout_error` per step | V-JEPA 2-AC (L1), DINO-WM and LeWM (MSE) |
| How good are decoded predictions? | `ssim`, `psnr` | DINO-WM |
| How close are two point sets? | `chamfer_distance` | DINO-WM (rope, granular) |
| Does retrieval find the right item? | `recall_at_k`, `topk_accuracy` | VL-JEPA (retrieval, zero-shot classification) |
| How often does the robot succeed, and how sure are we? | `success_rate` (with a Wilson interval) | DINO-WM, LeWM, V-JEPA 2-AC report the rate; the interval is added here |
| How fast does it get there? | `first_success` (steps to success) | time to goal, a common efficiency measure |
| How close does it get? | `progress` (fraction of the start distance closed) | partial credit for failed episodes |
| How direct is the path? | `spl` (success weighted by path length) | Anderson et al. 2018, standard in embodied navigation |
| How smooth are the commands? | `smoothness` (change between consecutive actions) | flags jittery commands from sampling planners |
| Push-T score | `PushT.coverage` (overlap with the goal region) | Diffusion Policy's Push-T score |

Protocols that run a model and then apply these metrics live in [`torch_world.evaluation`](../api/evaluation.md): `goal_reaching` for planners and `vqa` for question answering.

## Probing frozen features

```python
from torch_world import metrics

with torch.no_grad():
    z = encoder(images)                          # (N, D)
probe = metrics.linear_probe(z[:4000], y[:4000], z[4000:], y[4000:])
probe.score, probe.predictions                   # R² for float targets, accuracy for class ids
metrics.knn_accuracy(z[:4000], labels[:4000], z[4000:], labels[4000:], k=20)
metrics.rankme(z)                                # from 1 (collapsed) to D
```

Class probes fit multinomial logistic regression with L-BFGS; regression probes are solved in closed form. Both standardize features and apply a small L2 penalty, `ridge`. For video or patch features, `attentive_probe(tokens, ...)` trains a cross-attention pooler on `(N, T, D)` tokens instead, which can find the tokens that matter.

`lidar` needs several views of each sample, `(N, Q, D)`, for example the embeddings of `Q` crops of `N` images.

## Prediction quality of world models

```python
with torch.no_grad():
    real = model.encode(pixels)                  # (B, T, ...) encodings of the real frames
    imagined = model.rollout(real[:, :1], actions[:, : T - 1])
metrics.rollout_error(imagined, real)            # (T,), L1 per step; p=2 for MSE
```

Compare against the "nothing changes" baseline, `real[:, :1].expand_as(real)`: a useful model stays well below it. When latents can be decoded to images (`PixelPatches.unpatchify`, or a trained decoder), `ssim` and `psnr` score each predicted frame.

## Robotics and control

`evaluation.goal_reaching` records everything these measures need (states, per-step success, actions, planning time), and its result prints them as one table per policy:

```python
from torch_world import evaluation, plots

results = evaluation.goal_reaching(env, policies, start, goal, steps=30)
print(results)          # success [95% CI], reached, steps, final distance, progress, SPL, smoothness, ms / step
results.summary()       # the same, as a dict per policy
plots.bars({"Push-T": results.intervals()}, xlabel="success rate", baseline="random actions")
```

| measure | meaning |
|---|---|
| success | final success rate with its 95% Wilson interval: with 16 episodes, 50% means anything from 28% to 72%, so compare intervals, not points |
| reached | share of episodes that satisfied the goal at any step, even if they drifted away afterwards. OGBench ends an episode at the first success, so this is the column comparable to OGBench success rates |
| steps | mean steps to first reach the goal, over the episodes that did |
| final distance | median distance to the goal at the end (meters in the MuJoCo environments) |
| progress | mean fraction of the start distance closed: partial credit for failures |
| SPL | success weighted by path length of what the task moves (`env.object_position`): the cube on the table plane for pick-and-place, the gripper in 3D for reach. A detour twice too long counts one half. Push-T has no SPL: its goal includes an angle, which a path length cannot measure; coverage scores it instead |
| smoothness | mean change between consecutive actions; sampling planners like CEM tend to jitter |
| ms / step | wall time of one policy call, the planning cost LeWM compares against DINO-WM |
| coverage | for Push-T, the best overlap of the block with the goal T over the episode, the score of Diffusion Policy |

Environments provide `distance` and `success`; `object_position` (for SPL) and `coverage` are used when present.
