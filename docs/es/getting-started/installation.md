# Instalación

`torch_world` requiere Python 3.10 o posterior y PyTorch 2.1 o posterior. PyTorch es su única dependencia obligatoria.

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

## Extras opcionales {#optional-extras}

| extra | instala | para |
|---|---|---|
| `examples` | `matplotlib` | [`torch_world.plots`](../guides/plots.md) y los notebooks |
| `action` | `mujoco` | los entornos [`RobotCube`](../guides/environments.md#robotcube) y `PushT` |
| `ogbench` | `ogbench`, con Python 3.12 o anterior | [tareas de OGBench](../guides/benchmark-environments.md#ogbench) y [datasets](../datasets/trajectories.md#ogbench-datasets) |
| `dmcontrol` | `dm_control`, con Python 3.12 o anterior | [tareas de DeepMind Control Suite](../guides/benchmark-environments.md#external-suites) |
| `metaworld` | `metaworld` | [tareas de Meta-World](../guides/benchmark-environments.md#external-suites) |
| `maniskill` | `mani_skill`; en Linux, x86_64 con Python 3.12 o anterior | [tareas de ManiSkill](../guides/benchmark-environments.md#external-suites); requiere un driver de Vulkan |
| `myosuite` | `myosuite`, MuJoCo 3.7 o posterior | [tareas de MyoSuite](../guides/benchmark-environments.md#external-suites); no compatible con el extra `benchmark` |
| `robocasa` | `pyarrow`, `av` | leer [demostraciones de RoboCasa](../datasets/trajectories.md#robocasa-demonstrations); RoboCasa en sí se instala desde el código fuente |
| `robodojo` | `h5py` | leer [episodios de RoboDojo](../datasets/trajectories.md#robodojo-episodes) |
| `wam` | `diffusers`, `accelerate` | el [`VideoVAE`](../models/wam/dreamzero.md) preentrenado de Wan 2.1 de DreamZero |
| `test` | `pytest` | ejecutar la suite de tests |

```bash
pip install "torch-world[examples]"
```

## Comprobar la instalación {#check-the-install}

```python
import torch_world

print(torch_world.__version__)
```

::: tip GPU
Todos los módulos y entornos son PyTorch puro y se ejecutan en el dispositivo al que los muevas. Los entornos reciben un argumento `device` y renderizan los lotes directamente en él.
:::
