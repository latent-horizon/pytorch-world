# クイックスタート

このページでは、ピクセルから小さな世界モデルを学習し、それを使ってゴール画像に到達します。ノートパソコンの CPU で数分で終わります。

## 1. データを収集する {#1-collect-data}

`torch_world.envs` の環境はバッチ化されており、RGB 画像をレンダリングします。`collect` はランダムな行動を実行し、エピソードをテンソルとして返します。

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

[`TrajectoryDataset`](../api/datasets.md) は各エピソードを `window` ステップのスライディングウィンドウに切り分けます。

## 2. モデルを構築する {#2-build-a-model}

[`LeWM`](../models/jepa/lewm.md) には、1 枚の画像 `(N, C, H, W)` を 1 つの埋め込み `(N, D)` に写すエンコーダが必要です。[`CLSEncoder`](../api/nn.md#torch_world.nn.CLSEncoder) は、CLS トークンがまさにそれを行う ViT です。

```python
from torch_world.models import LeWM
from torch_world.nn import CLSEncoder

encoder = CLSEncoder(img_size=32, patch_size=8, embed_dim=64, depth=3, num_heads=4)
model = LeWM(encoder, action_dim=2, embed_dim=64, history=3, depth=3, num_heads=4, head_dim=16,
             mlp_dim=256, proj_dim=256, lamb=0.01)
```

## 3. 学習する {#3-train}

すべてのモデルは、スカラーの `"loss"` とその構成要素を含む dict を返します。

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

`recompute_batchnorm` は評価の前にプロジェクタの BatchNorm 統計量を再推定します。詳しくは[学習](../guides/training.md)を参照してください。

## 4. プランニングする {#4-plan}

[`CEM`](../guides/planning.md) は、想像された最終的な潜在表現がゴール画像の潜在表現に最も近くなる行動系列を探索します。閉ループでは、最初の行動を実行してから再びプランニングします。

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

## 次のステップ {#next}

- [コンセプト](concepts.md)：2 つの基底クラスとテンソル形状の規約。
- [モデル](../models/index.md)：各モデルが何を予測し、どう呼び出すか。
- [LeWM ノートブック](../examples/action/lewm.ipynb)：このページの内容を図、プローブ、結果付きで。
- [ガイド](../guides/index.md)：独自のモデル、環境、データセット、プランナーを持ち込む。
