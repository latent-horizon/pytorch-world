# Bring your own dataset

Models in `torch_world` take batches of tensors, not a dataset type. Any data works once it reaches the model as a dict of `(B, T, ...)` tensors, for example `model(batch["pixels"], batch["actions"])`.

| your data | use |
|---|---|
| equal-length episodes | a dict of tensors `(N, T, ...)` and [`TrajectoryDataset`](#episodes-in-memory) |
| episodes of different lengths | [cut them into chunks](#episodes-of-different-lengths) first |
| too large for memory, or already a `Dataset` | [your own `Dataset`](#your-own-dataset) that returns dicts of windows |

## Episodes in memory

Store each quantity as one tensor `(N, T, ...)`: `N` episodes of `T` steps, with the same `N` and `T` for every key.

| key | shape | convention |
|---|---|---|
| `pixels` | `(N, T, 3, H, W)` | float images in `[0, 1]` |
| `actions` | `(N, T, A)` | `actions[:, t]` is applied at step `t`; the last one is padding |
| `states`, or any other key | `(N, T, ...)` | proprioception, probe targets; cut like the rest |

```python
import numpy as np
import torch
from torch.utils.data import DataLoader

from torch_world.datasets import TrajectoryDataset

raw = np.load("episodes.npz")                                        # uint8 frames (N, T, H, W, 3), actions (N, T, A)
data = {
    "pixels": torch.from_numpy(raw["frames"]).permute(0, 1, 4, 2, 3).float() / 255,
    "actions": torch.from_numpy(raw["actions"]).float(),
}
dataset = TrajectoryDataset(data, window=model.history + 1)
loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
```

Each item is one window: every key cut over the same `window` steps. There are `N * (T - window + 1)` items. Scale actions to the range the planner samples, usually `[-1, 1]`.

With `frameskip=k`, a window keeps every `k`-th frame, and each of its steps concatenates the `k` actions that follow that frame, so actions `(N, T, A)` become `(window, k * A)`. LeWM and DINO-WM train this way. `stacked` names the keys that are concatenated rather than subsampled; the default is `("actions",)`.

```python
TrajectoryDataset(data, window=2, frameskip=5)[0]["actions"].shape     # (2, 5 * A)
```

## Episodes of different lengths

`TrajectoryDataset` needs a shared `T`. Cut longer episodes into chunks of one length so that no window crosses from one episode into the next:

```python
def chunk(episodes, length):
    """Cut variable-length episodes into equal chunks of ``length`` steps, dropping each remainder."""
    return {
        key: torch.cat([e[key][: len(e[key]) // length * length].unflatten(0, (-1, length)) for e in episodes])
        for key in episodes[0]
    }


episodes = [{"pixels": ..., "actions": ...}, ...]    # one dict of (T_i, ...) tensors per episode
dataset = TrajectoryDataset(chunk(episodes, 16), window=model.history + 1)
```

Data from a [benchmark environment](custom-environment.md) is already aligned: `collect` returns `observations` and `next_observations` with a `valid` mask that is `False` after an episode ends. Keep only valid transitions; never train across padding or reset boundaries.

## Your own Dataset

For data on disk, or any other source, write a `torch.utils.data.Dataset` whose items are dicts of `(window, ...)` tensors. The default `DataLoader` collation stacks them into `(B, window, ...)`, which is what the models expect.

```python
from torch.utils.data import Dataset


class Clips(Dataset):
    def __init__(self, paths, window):
        self.paths, self.window = paths, window

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        clip = torch.load(self.paths[i], weights_only=True)     # {"pixels": (T, 3, H, W), "actions": (T, A)}
        t = torch.randint(len(clip["actions"]) - self.window + 1, ()).item()
        return {key: value[t : t + self.window] for key, value in clip.items()}
```

## Checklist

| check | why |
|---|---|
| windows hold `model.history + 1` steps | one-step prediction needs the context and the next frame |
| images are float in `[0, 1]`, channels first | the built-in encoders assume it |
| actions use the planner's bounds | CEM samples in `bounds`, usually `[-1, 1]` |
| `actions[:, t]` follows frame `t` | the model predicts frame `t + 1` from frame `t` and `actions[:, t]` |
| tensors are on the training device before building `TrajectoryDataset` | windows are then cut on the GPU, without host-to-device copies |

The built-in datasets ([Trajectories](../datasets/trajectories.md), [MedMNIST](../datasets/medmnist.md), [Medical VQA](../datasets/medical-vqa.md)) follow the same conventions.
