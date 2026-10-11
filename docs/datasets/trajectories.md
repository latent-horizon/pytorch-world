# Trajectories

World models train on short windows of experience: `history + 1` frames and the actions between them. `TrajectoryDataset` cuts collected episodes into such windows.

## Collect episodes

Every [environment](../guides/environments.md) has a `collect` method that returns a batch of episodes as tensors that share their length `T`:

```python
from torch_world.envs import PointMass, RobotCube

data = PointMass(batch_size=2048, size=32).collect(16)       # random actions
data = RobotCube("ur10e", batch_size=256).collect(64)      # play data from a noisy expert
```

| key | shape | |
|---|---|---|
| `pixels` | `(N, T, 3, size, size)` | images in `[0, 1]` |
| `actions` | `(N, T, action_dim)` | `actions[:, t]` is applied at `states[:, t]`; the last one is padding |
| `states` | `(N, T, state_dim)` | the true state, for probes and proprioception |

Your own data works the same way: any dict of tensors `(N, T, ...)` with a shared `N` and `T`.

## Cut windows

```python
from torch.utils.data import DataLoader

from torch_world.datasets import TrajectoryDataset

dataset = TrajectoryDataset(data, window=4)
len(dataset)                  # N * (T - window + 1)
dataset[0]["pixels"].shape    # (4, 3, 32, 32)

loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
for batch in loader:
    out = model(batch["pixels"], batch["actions"])
```

Item `i` is window `i % (T - window + 1)` of episode `i // (T - window + 1)`, with every key cut the same way. Move the tensors to the GPU before building the dataset and the windows are cut there too, with no host-to-device copies during training.

## Choosing the window

| model | window |
|---|---|
| [LeWM](../models/jepa/lewm.md), [DINO-WM](../models/jepa/dinowm.md) | `model.history + 1` |
| a model with `history = 1`, like the [custom model](../guides/custom-model.md) | `2` |
| [RoboJEPA](../models/jepa/robojepa.md) | `rollout_steps + 1` or more; it trains on whole clips |

## Reference

[`TrajectoryDataset`](../api/datasets.md#torch_world.datasets.TrajectoryDataset) in the API reference.
