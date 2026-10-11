# Quick start

This page trains a small world model from pixels and uses it to reach goal images. It takes a few minutes on a laptop CPU.

## 1. Collect data

Environments in `torch_world.envs` are batched and render RGB images. `collect` plays random actions and returns episodes as tensors.

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

[`TrajectoryDataset`](../api/datasets.md) cuts every episode into sliding windows of `window` steps.

## 2. Build a model

[`LeWM`](../models/jepa/lewm.md) needs an encoder that maps one image `(N, C, H, W)` to one embedding `(N, D)`. [`CLSEncoder`](../api/nn.md#torch_world.nn.CLSEncoder) is a ViT whose CLS token does exactly that.

```python
from torch_world.models import LeWM
from torch_world.nn import CLSEncoder

encoder = CLSEncoder(img_size=32, patch_size=8, embed_dim=64, depth=3, num_heads=4)
model = LeWM(encoder, action_dim=2, embed_dim=64, history=3, depth=3, num_heads=4, head_dim=16,
             mlp_dim=256, proj_dim=256, lamb=0.01)
```

## 3. Train

Every model returns a dict with a scalar `"loss"` and its components.

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

`recompute_batchnorm` re-estimates the BatchNorm statistics of the projectors before evaluation; see [Training](../guides/training.md).

## 4. Plan

[`CEM`](../guides/planning.md) searches for action sequences whose imagined final latent is closest to the latent of a goal image. In closed loop, execute the first action and plan again.

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

## Next

- [Concepts](concepts.md): the two base classes and the shape conventions.
- [Models](../models/index.md): every model, what it predicts and how to call it.
- [LeWM notebook](../examples/action/lewm.ipynb): this page with figures, probes and results.
