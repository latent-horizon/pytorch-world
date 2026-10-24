# 安装

`torch_world` 需要 Python 3.10 或更高版本，以及 PyTorch 2.1 或更高版本。PyTorch 是它唯一的必需依赖。

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

## 可选依赖 {#optional-extras}

| extra | 安装内容 | 用途 |
|---|---|---|
| `examples` | `matplotlib` | [`torch_world.plots`](../guides/plots.md) 及示例 notebook |
| `action` | `mujoco` | [`RobotCube`](../guides/environments.md#robotcube) 和 `PushT` 环境 |
| `ogbench` | `ogbench`，需 Python 3.12 或更早版本 | [OGBench 任务](../guides/benchmark-environments.md#ogbench)与[数据集](../datasets/trajectories.md#ogbench-datasets) |
| `dmcontrol` | `dm_control`，需 Python 3.12 或更早版本 | [DeepMind Control Suite 任务](../guides/benchmark-environments.md#external-suites) |
| `metaworld` | `metaworld` | [Meta-World 任务](../guides/benchmark-environments.md#external-suites) |
| `maniskill` | `mani_skill`；在 Linux 上需 x86_64 及 Python 3.12 或更早版本 | [ManiSkill 任务](../guides/benchmark-environments.md#external-suites)；需要 Vulkan 驱动 |
| `myosuite` | `myosuite`，MuJoCo 3.7 或更高版本 | [MyoSuite 任务](../guides/benchmark-environments.md#external-suites)；不能与 `benchmark` extra 同时安装 |
| `robocasa` | `pyarrow`、`av` | 读取 [RoboCasa 演示数据](../datasets/trajectories.md#robocasa-demonstrations)；RoboCasa 本身需从源码安装 |
| `robodojo` | `h5py` | 读取 [RoboDojo episode](../datasets/trajectories.md#robodojo-episodes) |
| `wam` | `diffusers`、`accelerate` | DreamZero 使用的预训练 Wan 2.1 [`VideoVAE`](../models/wam/dreamzero.md) |
| `test` | `pytest` | 运行测试套件 |

```bash
pip install "torch-world[examples]"
```

## 检查安装 {#check-the-install}

```python
import torch_world

print(torch_world.__version__)
```

::: tip GPU
所有模块和环境都是纯 PyTorch 实现，可以运行在你将其移动到的任何设备上。环境接受 `device` 参数，并直接在该设备上批量渲染。
:::
