# Installation

`torch_world` needs Python 3.10 or newer and PyTorch 2.1 or newer. PyTorch is its only required dependency.

::: code-group

```bash [uv]
uv add torch_world
```

```bash [pip]
pip install torch-world
```

```bash [from source]
git clone <repository-url> pytorch-world
cd pytorch-world
uv sync
```
:::

## Optional extras

| extra | installs | for |
|---|---|---|
| `examples` | `matplotlib` | [`torch_world.plots`](../guides/plots.md) and the notebooks |
| `action` | `mujoco` | the [`RobotCube`](../guides/environments.md#robotcube) and `PushT` environments |
| `ogbench` | `ogbench`, on Python 3.12 or earlier | [OGBench tasks](../guides/benchmark-environments.md#ogbench) and [datasets](../datasets/trajectories.md#ogbench-datasets) |
| `dmcontrol` | `dm_control`, on Python 3.12 or earlier | [DeepMind Control Suite tasks](../guides/benchmark-environments.md#external-suites) |
| `metaworld` | `metaworld` | [Meta-World tasks](../guides/benchmark-environments.md#external-suites) |
| `maniskill` | `mani_skill`; on Linux, x86_64 with Python 3.12 or earlier | [ManiSkill tasks](../guides/benchmark-environments.md#external-suites); needs a Vulkan driver |
| `myosuite` | `myosuite`, MuJoCo 3.7 or later | [MyoSuite tasks](../guides/benchmark-environments.md#external-suites); not with the `benchmark` extra |
| `robocasa` | `pyarrow`, `av` | reading [RoboCasa demonstrations](../datasets/trajectories.md#robocasa-demonstrations); RoboCasa itself installs from source |
| `robodojo` | `h5py` | reading [RoboDojo episodes](../datasets/trajectories.md#robodojo-episodes) |
| `wam` | `diffusers`, `accelerate` | the pre-trained Wan 2.1 [`VideoVAE`](../models/wam/dreamzero.md) of DreamZero |
| `test` | `pytest` | running the test suite |

```bash
pip install "torch-world[examples]"
```

## Check the install

```python
import torch_world

print(torch_world.__version__)
```

::: tip GPU
Every module and environment is plain PyTorch and runs on whatever device you move it to. Environments take a `device` argument and render batches directly on it.
:::
