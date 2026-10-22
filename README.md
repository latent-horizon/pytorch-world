<p align="center">
  <img alt="pytorch-world" src="docs/assets/banner-sky.png">
</p>

<p align="center">
  <a href="https://pypi.org/project/torch-world/"><img alt="PyPI" src="https://img.shields.io/pypi/v/torch-world"></a>
  <a href="https://pypi.org/project/torch-world/"><img alt="Python" src="https://img.shields.io/badge/python-%E2%89%A53.10-blue"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/pytorch-%E2%89%A52.1-ee4c2c?logo=pytorch&logoColor=white"></a>
  <a href="https://github.com/latent-horizon/pytorch-world/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/latent-horizon/pytorch-world/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://latent-horizon.github.io/pytorch-world/"><img alt="Docs" src="https://github.com/latent-horizon/pytorch-world/actions/workflows/docs.yml/badge.svg"></a>
  <a href="LICENSE.md"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue"></a>
</p>

PyTorch World (`torch_world`) is a library of world models for PyTorch. It gives world-model research shared parts (layers, masking, objectives, planners, environments) so that a new model is a short file and existing models are easy to compare.

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
| `torch_world.nn` | `VisionTransformer` (image and video), `MaskedPredictor`, `Transformer`, `Block`, `AdaLNBlock`, `Attention`, `MLP`, sin-cos and rotary positions; ready encoders `CLSEncoder`, `PixelPatches` and `PatchTokens.from_hub` (DINOv2); `TextEncoder` and `WordTokenizer`; video autoencoders `VideoVAE` (Wan 2.1) and `ConvVideoAE`; flow-matching helpers |
| `torch_world.masks` | `MultiBlockMask` (2D), `MultiBlock3DMask` (video), `frame_causal_mask`, `apply_masks` |
| `torch_world.losses` | `SIGReg`, `info_nce` |
| `torch_world.models` | the `JEPA`, `WorldModel` and `WorldActionModel` base classes and the models below |
| `torch_world.planning` | `CEM`, for any `WorldModel`, scoring the final or every imagined step; `MPPI`, TD-MPC2's planner for reward; `WAMPlanner` and `DreamZeroPolicy` for world action models |
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
| `LeVJEPA` | [Kuhn et al., 2026](https://arxiv.org/abs/2608.27395) | [galilai-group/levjepa](https://github.com/galilai-group/levjepa) | [vision/levjepa](examples/vision/levjepa.ipynb) |
| `VLJEPA` | [Chen et al., 2026](https://arxiv.org/abs/2512.10942) | not released | [vqa/vljepa](examples/vqa/vljepa.ipynb) |
| `LeWAM` | [Fu et al., 2026](https://arxiv.org/abs/2610.12818) | [MinghaoFu/lewam](https://github.com/MinghaoFu/lewam) | [action/lewam](examples/action/lewam.ipynb) |
| `LeWM` | [Maes et al., 2026](https://arxiv.org/abs/2603.19312) | [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) | [action/lewm](examples/action/lewm.ipynb) |
| `RoboJEPA` | [Zholus et al., 2026](https://arxiv.org/abs/2610.10515) | [facebookresearch/robo_jepa](https://github.com/facebookresearch/robo_jepa) | [action/robojepa](examples/action/robojepa.ipynb) |

Joint-embedding predictive architectures (JEPAs) predict the *representation* of a target from the representation of a context, instead of reconstructing pixels. Action-conditioned JEPAs are world models: they predict the next latent state from the current one and an action, and planning searches for the actions that bring the imagined latent closest to a goal.

### DINO

| model | paper | reference code | notebook |
|---|---|---|---|
| `DINOv2` | [Oquab et al., 2024](https://arxiv.org/abs/2304.07193) | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | [vision/dinov2](examples/vision/dinov2.ipynb) |
| `DINOWM` | [Zhou et al., 2025](https://arxiv.org/abs/2411.04983) | [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm) | [action/dinowm](examples/action/dinowm.ipynb) |

Self-distillation: a student matches an EMA teacher's assignments of image crops to learned prototypes, on [cls] tokens (DINO) and masked patches (iBOT). The notebook trains it next to LeVJEPA on the same photos and footage. DINO-WM learns latent dynamics over the patch features of a frozen DINOv2 encoder and plans with CEM.

### TD-MPC

| model | paper | reference code | notebook |
|---|---|---|---|
| `TDMPC2` | [Hansen et al., 2024](https://arxiv.org/abs/2310.16828) | [nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2) | [action/tdmpc2](examples/action/tdmpc2.ipynb) |

TD-MPC world models have no decoder: latent dynamics are trained to predict the next latent, the reward and, by temporal-difference learning, the value. MPPI plans for the highest imagined return, with proposals from a learned policy prior.

### World action models

| model | paper | reference code | notebook |
|---|---|---|---|
| `DreamZero` | [Ye et al., 2026](https://arxiv.org/abs/2602.15922) | not used | [action/dreamzero](examples/action/dreamzero.ipynb) |

World action models denoise future video and the actions that produce it together, so the model is both the policy and the world model. DreamZero is an autoregressive joint video-action diffusion transformer on Wan 2.1 video-VAE latents (`pip install torch-world[wam]`); its notebook trains it from scratch on a UR10e arm and compares it with LeWAM.

## Environment benchmarks

Work with **190 environments**: 33 built-in tasks (28 new tasks and five benchmark recipes for the original environments), 13 from OGBench, 52 from the DeepMind Control Suite, Meta-World, ManiSkill and MyoSuite, 50 from RoboCasa and 42 from RoboDojo. Discover the registered tasks with `torch_world.envs.list_envs()` and create them with `make(id)`: planning puzzles, MuJoCo manipulation and control, Newton articulated dynamics, Isaac Lab robotics, continuous control, musculoskeletal control and kitchen tasks. RoboDojo's dual-arm tasks are evaluated by RoboDojo's own client, which a torch-world policy serves through XPolicyLab. [Setup and contributor API](docs/guides/benchmark-environments.md).

```python
from torch_world.envs import KeyDoor
from torch_world.benchmarks import run_experiment

world = KeyDoor()
world.show_env()  # initial state and GT
result = run_experiment(world, output_dir="runs/benchmarks")
world.show_success()
world.close()
```

## Adding a model

Subclass `JEPA` (representation learning) or `WorldModel` (action-conditioned dynamics) and build it from the shared modules. A `WorldModel` that implements `encode`, `predict` and `forward` gets `rollout`, `cost` and `CEM` planning with no extra code. See [examples/custom_model.ipynb](examples/custom_model.ipynb) and [CONTRIBUTING.md](CONTRIBUTING.md).

## Tests

```bash
pytest
```

## Contributors

<a href="https://github.com/latent-horizon/pytorch-world/graphs/contributors">
  <img alt="Contributors" src="https://contrib.rocks/image?repo=latent-horizon/pytorch-world">
</a>

Contributions are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md).

## Supported by

<a href="https://github.com/latent-horizon">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/latent-horizon/logo-cream.svg">
    <img alt="Latent Horizon" src="docs/assets/latent-horizon/logo-black.svg" height="40">
  </picture>
</a>

## License

Apache 2.0
