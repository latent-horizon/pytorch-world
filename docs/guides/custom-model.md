# Bring your own model

A new model is a subclass of `WorldModel` (action-conditioned dynamics) or `JEPA` (representation learning), built from the shared modules. This page writes a world model in about 30 lines, then trains it, plans with it and benchmarks it with no further code. The [custom model notebook](../examples/custom_model.ipynb) runs it end to end.

## The contract

For a `WorldModel`, implement three methods:

| method | maps |
|---|---|
| `encode(pixels, ...)` | observations `(B, T, ...)` to latents `(B, T, ...)` |
| `predict(z, actions, ...)` | `z[:, t]` and `actions[:, t]` to a prediction of `z[:, t + 1]`, for every `t` |
| `forward(...)` | a batch to `{"loss": ..., **components}` |

and set `history`, the number of past latents `predict` should see during rollouts. `rollout` and `cost` are inherited, so [planners](custom-planner.md) work as is. Override `cost` for another distance to the goal.

A `JEPA` only needs `forward`. For an EMA target encoder, build it with `make_target_encoder` and call `update_target` after each step ([Training](training.md#ema-targets)).

## A convolutional JEPA

A convolutional encoder, residual MLP dynamics and SIGReg against collapse: a minimal LeWM-style model.

```python
import torch
from torch import nn

from torch_world.losses import SIGReg
from torch_world.models import WorldModel
from torch_world.nn import MLP


class ConvJEPA(WorldModel):
    history = 1

    def __init__(self, action_dim: int, dim: int = 32, lamb: float = 0.01):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.GELU(),
            nn.Flatten(), nn.Linear(64 * 4 * 4, dim),
        )
        self.dynamics = MLP([dim + action_dim, 256, 256, dim])
        self.sigreg = SIGReg(256)
        self.lamb = lamb

    def encode(self, pixels):
        return self.encoder(pixels.flatten(0, 1)).unflatten(0, pixels.shape[:2])

    def predict(self, z, actions):
        return z + self.dynamics(torch.cat([z, actions], dim=-1))

    def forward(self, pixels, actions):
        z = self.encode(pixels)
        pred = (self.predict(z[:, :-1], actions[:, :-1]) - z[:, 1:]).square().mean()
        sigreg = self.sigreg(z.transpose(0, 1))
        return {"loss": pred + self.lamb * sigreg, "prediction": pred, "SIGReg": sigreg}
```

Train it like any other model ([Training](training.md)), and plan with [`CEM`](planning.md):

```python
planner = CEM(horizon=5, action_dim=2, num_samples=512, num_elites=48, iterations=10, bounds=(-1, 1))
actions = planner(model, model.encode(obs[:, None]), goal)
```

## Benchmark it

`run_experiment` takes a factory that receives the observation shape and the action dimension. The one-step benchmark protocol requires `history = 1`. The encoder above expects 32 × 32 images:

```python
from torch_world.benchmarks import run_experiment, show_results
from torch_world.envs import Maze

with Maze(size=32) as world:
    result = run_experiment(world, model_factory=lambda shape, action_dim: ConvJEPA(action_dim))
show_results(result)
```

For other shapes, start from `benchmarks.observation_encoder(shape)`: a small CNN for images, an MLP for vectors. The [custom model notebook](../examples/envs/custom_model_three_envs.ipynb) compares such a model with LeWM on three tasks.

## Parts to reach for

| need | use |
|---|---|
| ready image encoders | `nn.CLSEncoder` (one embedding per image), `nn.PixelPatches` (raw patches), `nn.PatchTokens.from_hub` (frozen DINOv2) |
| image or video encoder | `nn.VisionTransformer` (`from_size("small")`, `num_frames`, `use_rope`, `cls_token`) |
| text | `nn.TextEncoder`, `nn.WordTokenizer` |
| evaluation | `evaluation.goal_reaching`, `evaluation.vqa`; the metrics of the papers in `metrics` (see [Metrics](metrics.md)) |
| predictor over masked positions | `nn.MaskedPredictor` |
| action-conditioned transformer | `nn.Transformer(conditional=True)` (adaLN-zero blocks) |
| causal over frames | `masks.frame_causal_mask`, or `is_causal=True` with one token per frame |
| anti-collapse | `losses.SIGReg`, or an EMA target via `JEPA.make_target_encoder` |
| contrastive alignment | `losses.info_nce` |
| initialization as in the JEPA code | `nn.init_weights`, `nn.rescale_blocks` |

## Adding it to the library

Models live in `torch_world/models/jepa/`, one file each, exported from `torch_world.models`. Each comes with tests in `tests/models/` and a notebook in `examples/`. See [Contributing](../contributing.md).
