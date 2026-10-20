# 快速开始

本页从像素训练一个小型世界模型，并用它到达目标图像。在笔记本电脑 CPU 上只需几分钟。

## 1. 收集数据 {#1-collect-data}

`torch_world.envs` 中的环境是批量化的，并渲染 RGB 图像。`collect` 执行随机动作，并以张量形式返回 episode。

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

[`TrajectoryDataset`](../api/datasets.md) 将每个 episode 切分为长度为 `window` 步的滑动窗口。

## 2. 构建模型 {#2-build-a-model}

[`LeWM`](../models/jepa/lewm.md) 需要一个编码器，将单张图像 `(N, C, H, W)` 映射为一个嵌入 `(N, D)`。[`CLSEncoder`](../api/nn.md#torch_world.nn.CLSEncoder) 是一个 ViT，其 CLS token 正好完成这一映射。

```python
from torch_world.models import LeWM
from torch_world.nn import CLSEncoder

encoder = CLSEncoder(img_size=32, patch_size=8, embed_dim=64, depth=3, num_heads=4)
model = LeWM(encoder, action_dim=2, embed_dim=64, history=3, depth=3, num_heads=4, head_dim=16,
             mlp_dim=256, proj_dim=256, lamb=0.01)
```

## 3. 训练 {#3-train}

每个模型都返回一个 dict，包含标量 `"loss"` 及其各个分量。

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

`recompute_batchnorm` 在评估前重新估计投影头的 BatchNorm 统计量；参见[训练](../guides/training.md)。

## 4. 规划 {#4-plan}

[`CEM`](../guides/planning.md) 搜索动作序列，使其想象出的最终隐状态最接近目标图像的隐状态。在闭环中，执行第一个动作后重新规划。

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

## 下一步 {#next}

- [核心概念](concepts.md)：两个基类与形状约定。
- [模型](../models/index.md)：所有模型、它们预测什么以及如何调用。
- [LeWM notebook](../examples/action/lewm.ipynb)：本页内容，附带图表、探针与结果。
- [指南](../guides/index.md)：接入你自己的模型、环境、数据集或规划器。
