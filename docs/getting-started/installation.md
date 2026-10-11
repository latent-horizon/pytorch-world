# Installation

`torch_world` needs Python 3.10 or newer and PyTorch 2.1 or newer. PyTorch is its only required dependency.

=== "uv"

    ```bash
    uv add torch_world
    ```

=== "pip"

    ```bash
    pip install torch-world
    ```

=== "from source"

    ```bash
    git clone <repository-url> pytorch-world
    cd pytorch-world
    uv sync
    ```

## Optional extras

| extra | installs | for |
|---|---|---|
| `examples` | `matplotlib` | [`torch_world.plots`](../guides/plots.md) and the notebooks |
| `robotics` | `mujoco` | the [`RobotCube`](../guides/environments.md#robotcube) environment |
| `test` | `pytest` | running the test suite |

```bash
pip install "torch-world[examples]"
```

## Check the install

```python
import torch_world

print(torch_world.__version__)
```

!!! tip "GPU"
    Every module and environment is plain PyTorch and runs on whatever device you move it to. Environments take a `device` argument and render batches directly on it.
