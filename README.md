<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.png">
    <img alt="pytorch-world" src="docs/assets/banner-light.png">
  </picture>
</p>

PyTorch World (`torch_world`) is a library of world models for PyTorch. Like PyTorch Geometric did for graph neural networks, it gives world-model research shared parts (layers, masking, objectives, planners, environments) so that a new model is a short file and existing models are easy to compare.

```bash
uv add torch_world                 # core: depends only on torch
uv add "torch_world[examples]"     # + matplotlib, for the notebooks' figures
uv add "torch_world[action]"       # + mujoco, for the RobotCube arms and Push-T
```

## Quick start

```python
from torch_world.envs import PointMass
from torch_world.models import LeWM
from torch_world.nn import CLSEncoder
from torch_world.planning import CEM

model = LeWM(CLSEncoder(img_size=32, patch_size=8, embed_dim=64), action_dim=2, embed_dim=64)
episodes = PointMass(batch_size=64).collect(4)                 # pixels (64, 4, 3, 32, 32), actions (64, 4, 2)
loss = model(episodes["pixels"], episodes["actions"])["loss"]  # prediction + SIGReg

model.eval()
z, goal = model.encode(episodes["pixels"][:, :1]), model.encode(episodes["pixels"][:, -1:])[:, 0]
actions = CEM(horizon=5, action_dim=2)(model, z, goal)         # (64, 5, 2)
```

## Structure

| module | contents |
|---|---|
| `torch_world.nn` | `VisionTransformer` (image and video), `MaskedPredictor`, `Transformer`, `Block`, `AdaLNBlock`, `Attention`, `MLP`, sin-cos and rotary positions; ready encoders `CLSEncoder`, `PixelPatches` and `PatchTokens.from_hub` (DINOv2); `TextEncoder` and `WordTokenizer` |
| `torch_world.masks` | `MultiBlockMask` (2D), `MultiBlock3DMask` (video), `frame_causal_mask`, `apply_masks` |
| `torch_world.losses` | `SIGReg`, `info_nce` |
| `torch_world.models` | the `JEPA` and `WorldModel` base classes and the models below |
| `torch_world.planning` | `CEM`, for any `WorldModel`, scoring the final or every imagined step |
| `torch_world.envs` | MuJoCo worlds: `RobotCube` (reach or pick-and-place with a UR10e or SO-101 arm, after OGBench `cube-single`) and `PushT`; the batched pixel worlds `Macaw` and `PointMass`; `run_policy` |
| `torch_world.datasets` | `TrajectoryDataset`, sliding windows over episodes; `MedMNIST` and `MedicalVQA`, templated medical VQA in the style of MMBU |
| `torch_world.evaluation` | `goal_reaching` for planners in any environment, with success rates and recorded videos; `vqa` and `compare` for retrieval question answering, with a shuffled-image control |
| `torch_world.metrics` | the metrics of the papers: linear, k-NN and attentive probes, RankMe, LiDAR, rollout error, SSIM, PSNR, Chamfer distance, recall@k; for robotics, success rates with confidence intervals, steps to success, progress, SPL and smoothness |
| `torch_world.utils` | `ema_update`, `linear_schedule`, `recompute_batchnorm` |
| `torch_world.plots` | matplotlib figures for notebooks (optional dependency) |

## Models

### JEPA

| model | paper | reference code | notebook |
|---|---|---|---|
| `IJEPA` | [Assran et al., 2023](https://arxiv.org/abs/2301.08243) | [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) | [vision/ijepa](examples/vision/ijepa.ipynb) |
| `VJEPA` (and V-JEPA 2) | [Bardes et al., 2024](https://arxiv.org/abs/2404.08471), [Assran et al., 2025](https://arxiv.org/abs/2506.09985) | [facebookresearch/jepa](https://github.com/facebookresearch/jepa), [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) | [vision/vjepa](examples/vision/vjepa.ipynb) |
| `LeJEPA` | [Balestriero & LeCun, 2025](https://arxiv.org/abs/2511.08544) | [galilai-group/lejepa](https://github.com/galilai-group/lejepa) | [vision/lejepa](examples/vision/lejepa.ipynb) |
| `VLJEPA` | [Chen et al., 2026](https://arxiv.org/abs/2512.10942) | not released | [vqa/vljepa](examples/vqa/vljepa.ipynb) |
| `LeWM` | [Maes et al., 2026](https://arxiv.org/abs/2603.19312) | [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) | [action/lewm](examples/action/lewm.ipynb) |
| `DINOWM` | [Zhou et al., 2025](https://arxiv.org/abs/2411.04983) | [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm) | [action/dinowm](examples/action/dinowm.ipynb) |
| `RoboJEPA` | [Zholus et al., 2026](https://arxiv.org/abs/2610.10515) | [facebookresearch/robo_jepa](https://github.com/facebookresearch/robo_jepa) | [action/robojepa](examples/action/robojepa.ipynb) |

Joint-embedding predictive architectures (JEPAs) predict the *representation* of a target from the representation of a context, instead of reconstructing pixels. Action-conditioned JEPAs are world models: they predict the next latent state from the current one and an action, and planning searches for the actions that bring the imagined latent closest to a goal.

## Adding a model

Subclass `JEPA` (representation learning) or `WorldModel` (action-conditioned dynamics) and build it from the shared modules. A `WorldModel` that implements `encode`, `predict` and `forward` gets `rollout`, `cost` and `CEM` planning with no extra code. See [examples/custom_model.ipynb](examples/custom_model.ipynb) and [CONTRIBUTING.md](CONTRIBUTING.md).

## Tests

```bash
pytest
```

## License

Apache 2.0
