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

## OGBench datasets

`load_ogbench` reads an [OGBench](https://github.com/seohongpark/ogbench) dataset into the same layout, downloading it with OGBench into `~/.ogbench/data` on first use. It needs the `ogbench` extra.

```python
from torch_world.datasets import TrajectoryDataset, load_ogbench

data = load_ogbench("visual-cube-single-play-v0")      # split="val" for the validation file
data["observations"].shape                             # (N, T, 3, 64, 64), uint8
dataset = TrajectoryDataset(data, window=4, frameskip=5)
```

| key | shape | |
|---|---|---|
| `observations` | `(N, T, D)`, or `(N, T, 3, 64, 64)` | float32 state vectors, or `visual-` images kept in uint8 to save memory: divide by 255 for `[0, 1]` |
| `actions` | `(N, T, action_dim)` | as above, the last one has no next frame |
| `qpos`, `qvel`, `button_states` | `(N, T, ...)` | with `info=True`, where the dataset records them |

Every OGBench dataset stores episodes of one length; a file that does not is rejected rather than padded. Single-task names are relabelled goal-conditioned data, so load their base name. The matching environments are [`OGBench`](../guides/benchmark-environments.md#ogbench).

## RoboDojo episodes

`load_robodojo` reads one [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) demonstration, an HDF5 file in XPolicyLab's trajectory format (download them with RoboDojo's `scripts/RoboDojo/download_data.sh huggingface hdf5`). It needs the `robodojo` extra, and XPolicyLab on the Python path when the frames are JPEG-encoded, since only its `decode_image_bit` decodes both of its byte formats correctly.

```python
from pathlib import Path

from torch.utils.data import ConcatDataset

from torch_world.datasets import TrajectoryDataset, load_robodojo

paths = sorted(Path("data/RoboDojo/stack_blocks").glob("*.hdf5"))
dataset = ConcatDataset([TrajectoryDataset(load_robodojo(p, size=64), window=4) for p in paths])
```

| key | shape | |
|---|---|---|
| `observations` | `(1, T, 3, size, size)` | uint8 frames of `camera` (`"cam_head"`), resized whole, so 640 x 480 is squeezed to a square |
| `states` | `(1, T, 14)` | `[left arm, left gripper, right arm, right gripper]`, joint angles in radians and grippers in `[0, 1]` |
| `actions` | `(1, T, 14)` | the commanded targets, packed the same way |

`action_type="ee"` packs 7-value end-effector poses `[x, y, z, qw, qx, qy, qz]` instead of joints. Episodes differ in length, so each file is its own `(1, T, ...)` episode. [`RoboDojoPolicy`](../guides/benchmark-environments.md#robodojo) resizes frames and packs actions the same way at evaluation.

## RoboCasa demonstrations

`load_robocasa` reads one episode of a [RoboCasa](https://robocasa.ai) dataset, a LeRobot directory as `python -m robocasa.scripts.download_datasets` extracts it, and reorders its actions to the order [`RoboCasa`](../guides/benchmark-environments.md#robocasa) environments take. It needs the `robocasa` extra (`pyarrow` and `av`), not RoboCasa itself.

```python
import json
from pathlib import Path

from torch.utils.data import ConcatDataset

from torch_world.datasets import TrajectoryDataset, load_robocasa

root = Path("datasets/v1.0/pretrain/atomic/OpenDrawer/20250819/lerobot")
episodes = json.loads((root / "meta" / "info.json").read_text())["total_episodes"]
dataset = ConcatDataset([TrajectoryDataset(load_robocasa(root, i, size=128), window=4) for i in range(episodes)])
```

| key | shape | |
|---|---|---|
| `observations` | `(1, T, 3, size, size)` | uint8 frames of `camera` (`"robot0_agentview_left"`; also `"robot0_agentview_right"`, `"robot0_eye_in_hand"`) |
| `states` | `(1, T, 16)` | base position and rotation, end-effector pose relative to the base, gripper joints |
| `actions` | `(1, T, 12)` | in the environment's order: end-effector position and rotation deltas, gripper, base motion, torso, mode |

`load_robocasa_scene(root, episode)` returns the demonstration's kitchen and initial simulator state, for `RoboCasa(...).reset(scenes=[...])`. RoboCasa's registry counts 100 human demonstrations per task in a pretraining dataset and 500 in a target dataset; `meta/info.json` has the exact number.

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
