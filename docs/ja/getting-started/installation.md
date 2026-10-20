# インストール

`torch_world` には Python 3.10 以上と PyTorch 2.1 以上が必要です。必須の依存関係は PyTorch のみです。

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

## オプションの追加依存 {#optional-extras}

| extra | インストールされるもの | 用途 |
|---|---|---|
| `examples` | `matplotlib` | [`torch_world.plots`](../guides/plots.md) とノートブック |
| `robotics` | `mujoco` | [`RobotCube`](../guides/environments.md#robotcube) 環境 |
| `ogbench` | `ogbench`（Python 3.12 以下） | [OGBench タスク](../guides/benchmark-environments.md#ogbench)と[データセット](../datasets/trajectories.md#ogbench-datasets) |
| `dmcontrol` | `dm_control`（Python 3.12 以下） | [DeepMind Control Suite タスク](../guides/benchmark-environments.md#external-suites) |
| `metaworld` | `metaworld` | [Meta-World タスク](../guides/benchmark-environments.md#external-suites) |
| `maniskill` | `mani_skill`（Linux では x86_64 かつ Python 3.12 以下） | [ManiSkill タスク](../guides/benchmark-environments.md#external-suites)。Vulkan ドライバが必要 |
| `myosuite` | `myosuite`、MuJoCo 3.7 以上 | [MyoSuite タスク](../guides/benchmark-environments.md#external-suites)。`benchmark` extra とは併用不可 |
| `robocasa` | `pyarrow`, `av` | [RoboCasa のデモンストレーション](../datasets/trajectories.md#robocasa-demonstrations)の読み込み。RoboCasa 本体はソースからインストール |
| `robodojo` | `h5py` | [RoboDojo のエピソード](../datasets/trajectories.md#robodojo-episodes)の読み込み |
| `test` | `pytest` | テストスイートの実行 |

```bash
pip install "torch-world[examples]"
```

## インストールの確認 {#check-the-install}

```python
import torch_world

print(torch_world.__version__)
```

::: tip GPU
すべてのモジュールと環境は素の PyTorch で書かれており、移動させた任意のデバイス上で動作します。環境は `device` 引数を受け取り、バッチをそのデバイス上で直接レンダリングします。
:::
