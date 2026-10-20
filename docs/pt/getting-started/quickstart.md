# Início rápido

Esta página treina um pequeno world model a partir de pixels e o usa para alcançar goal images. Leva poucos minutos na CPU de um laptop.

## 1. Colete dados {#1-collect-data}

Os environments em `torch_world.envs` operam em batch e renderizam imagens RGB. `collect` executa random actions e retorna episodes como tensors.

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

[`TrajectoryDataset`](../api/datasets.md) corta cada episode em sliding windows de `window` steps.

## 2. Construa um modelo {#2-build-a-model}

[`LeWM`](../models/jepa/lewm.md) precisa de um encoder que mapeie uma imagem `(N, C, H, W)` para um embedding `(N, D)`. [`CLSEncoder`](../api/nn.md#torch_world.nn.CLSEncoder) é um ViT cujo CLS token faz exatamente isso.

```python
from torch_world.models import LeWM
from torch_world.nn import CLSEncoder

encoder = CLSEncoder(img_size=32, patch_size=8, embed_dim=64, depth=3, num_heads=4)
model = LeWM(encoder, action_dim=2, embed_dim=64, history=3, depth=3, num_heads=4, head_dim=16,
             mlp_dim=256, proj_dim=256, lamb=0.01)
```

## 3. Treine {#3-train}

Todo modelo retorna um dict com um `"loss"` escalar e seus componentes.

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

`recompute_batchnorm` reestima as estatísticas de BatchNorm dos projectors antes da evaluation; veja [Training](../guides/training.md).

## 4. Planeje {#4-plan}

[`CEM`](../guides/planning.md) busca sequências de actions cujo latent final imaginado fique o mais próximo possível do latent de uma goal image. Em closed loop, execute a primeira action e planeje novamente.

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

## Próximos passos {#next}

- [Conceitos](concepts.md): as duas classes base e as convenções de shape.
- [Modelos](../models/index.md): cada modelo, o que ele prevê e como chamá-lo.
- [Notebook do LeWM](../examples/action/lewm.ipynb): esta página com figuras, probes e resultados.
- [Guias](../guides/index.md): traga seu próprio modelo, environment, dataset ou planner.
