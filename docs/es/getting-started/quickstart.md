# Inicio rápido

Esta página entrena un pequeño world model a partir de píxeles y lo usa para alcanzar imágenes objetivo. Tarda unos minutos en la CPU de un portátil.

## 1. Recopilar datos {#1-collect-data}

Los entornos de `torch_world.envs` funcionan por lotes y renderizan imágenes RGB. `collect` ejecuta acciones aleatorias y devuelve los episodios como tensores.

```python
import torch
from torch.utils.data import DataLoader

from torch_world.datasets import TrajectoryDataset
from torch_world.envs import PointMass

torch.manual_seed(0)
data = PointMass(batch_size=2048, size=32).collect(16)
# data["pixels"]  (2048, 16, 3, 32, 32)
# data["actions"] (2048, 16, 2)   actions[:, t] is applied at states[:, t]
# data["states"]  (2048, 16, 2)

loader = DataLoader(TrajectoryDataset(data, window=4), batch_size=128, shuffle=True, drop_last=True)
```

[`TrajectoryDataset`](../api/datasets.md) divide cada episodio en ventanas deslizantes de `window` pasos.

## 2. Construir un modelo {#2-build-a-model}

[`LeWM`](../models/jepa/lewm.md) necesita un encoder que convierta una imagen `(N, C, H, W)` en un embedding `(N, D)`. [`CLSEncoder`](../api/nn.md#torch_world.nn.CLSEncoder) es un ViT cuyo token CLS hace exactamente eso.

```python
from torch_world.models import LeWM
from torch_world.nn import CLSEncoder

encoder = CLSEncoder(img_size=32, patch_size=8, embed_dim=64, depth=3, num_heads=4)
model = LeWM(encoder, action_dim=2, embed_dim=64, history=3, depth=3, num_heads=4, head_dim=16,
             mlp_dim=256, proj_dim=256, lamb=0.01)
```

## 3. Entrenar {#3-train}

Todos los modelos devuelven un dict con un `"loss"` escalar y sus componentes.

```python
import itertools

from torch_world.utils import recompute_batchnorm

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
for epoch in range(40):
    for batch in loader:
        out = model(batch["pixels"], batch["actions"])  # {"loss", "pred_loss", "sigreg_loss"}
        opt.zero_grad()
        out["loss"].backward()
        opt.step()

recompute_batchnorm(model, itertools.islice(loader, 50), lambda b: model(b["pixels"], b["actions"]))
model.eval()
```

`recompute_batchnorm` vuelve a estimar las estadísticas de BatchNorm de los proyectores antes de la evaluación; consulta [Entrenamiento](../guides/training.md).

## 4. Planificar {#4-plan}

[`CEM`](../guides/planning.md) busca secuencias de acciones cuyo latente final imaginado esté lo más cerca posible del latente de una imagen objetivo. En lazo cerrado, ejecuta la primera acción y vuelve a planificar.

```python
from torch_world.envs import run_policy
from torch_world.planning import CEM

env = PointMass(batch_size=64, size=32)
env.reset()
goals = env.sample_goals((0.2, 0.5))
with torch.no_grad():
    goal = model.encode(env.render(goals)[:, None])[:, 0]

planner = CEM(horizon=5, action_dim=2, num_samples=512, num_elites=48, iterations=10, bounds=(-1, 1))
paths = run_policy(env, lambda obs: planner(model, model.encode(obs[:, None]), goal)[:, 0], steps=20)

success = ((paths[:, -1] - goals).norm(dim=-1) < 0.1).float().mean()
```

## Siguientes pasos {#next}

- [Conceptos](concepts.md): las dos clases base y las convenciones de formas.
- [Modelos](../models/index.md): cada modelo, qué predice y cómo llamarlo.
- [Notebook de LeWM](../examples/action/lewm.ipynb): esta página con figuras, probes y resultados.
- [Guías](../guides/index.md): trae tu propio modelo, entorno, dataset o planificador.
