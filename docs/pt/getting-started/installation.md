# Instalação

`torch_world` requer Python 3.10 ou mais recente e PyTorch 2.1 ou mais recente. PyTorch é sua única dependência obrigatória.

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

## Extras opcionais {#optional-extras}

| extra | instala | para |
|---|---|---|
| `examples` | `matplotlib` | [`torch_world.plots`](../guides/plots.md) e os notebooks |
| `action` | `mujoco` | os environments [`RobotCube`](../guides/environments.md#robotcube) e `PushT` |
| `ogbench` | `ogbench`, no Python 3.12 ou anterior | [tarefas do OGBench](../guides/benchmark-environments.md#ogbench) e [datasets](../datasets/trajectories.md#ogbench-datasets) |
| `dmcontrol` | `dm_control`, no Python 3.12 ou anterior | [tarefas do DeepMind Control Suite](../guides/benchmark-environments.md#external-suites) |
| `metaworld` | `metaworld` | [tarefas do Meta-World](../guides/benchmark-environments.md#external-suites) |
| `maniskill` | `mani_skill`; no Linux, x86_64 com Python 3.12 ou anterior | [tarefas do ManiSkill](../guides/benchmark-environments.md#external-suites); requer um driver Vulkan |
| `myosuite` | `myosuite`, MuJoCo 3.7 ou posterior | [tarefas do MyoSuite](../guides/benchmark-environments.md#external-suites); não funciona junto com o extra `benchmark` |
| `robocasa` | `pyarrow`, `av` | leitura de [demonstrações do RoboCasa](../datasets/trajectories.md#robocasa-demonstrations); o próprio RoboCasa é instalado a partir do código-fonte |
| `robodojo` | `h5py` | leitura de [episodes do RoboDojo](../datasets/trajectories.md#robodojo-episodes) |
| `wam` | `diffusers`, `accelerate` | o [`VideoVAE`](../models/wam/dreamzero.md) pretrained do Wan 2.1 usado pelo DreamZero |
| `test` | `pytest` | execução da test suite |

```bash
pip install "torch-world[examples]"
```

## Verifique a instalação {#check-the-install}

```python
import torch_world

print(torch_world.__version__)
```

::: tip GPU
Todo módulo e environment é PyTorch puro e roda no device para o qual você o mover. Os environments recebem um argumento `device` e renderizam os batches diretamente nele.
:::
