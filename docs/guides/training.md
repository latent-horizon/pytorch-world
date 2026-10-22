# Training

`torch_world` has no trainer. Every model is an `nn.Module` whose `forward` returns a dict with a scalar `"loss"`, so a training loop is the usual five lines.

```python
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for batch in loader:
    out = model(batch["pixels"], batch["actions"])
    opt.zero_grad()
    out["loss"].backward()
    opt.step()
```

The other keys of `out` are the loss components, ready to log.

## Data from episodes

[`TrajectoryDataset`](../api/datasets.md) turns episodes `(N, T, ...)` into sliding windows of `window` steps, one dict per window. World models usually train on `history + 1` frames.

```python
from torch_world.datasets import TrajectoryDataset

data = PointMass(batch_size=2048).collect(16)
dataset = TrajectoryDataset(data, window=model.history + 1)
```

## EMA targets

I-JEPA and V-JEPA predict the features of a target encoder: a frozen copy of the encoder, updated by exponential moving average after every optimizer step. The momentum usually rises linearly to 1.

```python
from torch_world.utils import linear_schedule

for step in range(steps):
    loss = model(images, masks_enc, masks_pred)["loss"]
    opt.zero_grad()
    loss.backward()
    opt.step()
    model.update_target(linear_schedule(0.996, 1.0, step, steps))
```

Your own `JEPA` subclass gets the same machinery: build the copy with `self.target_encoder = self.make_target_encoder(encoder)` and call `update_target`. For other modules, use [`ema_update`](../api/utils.md) directly.

## BatchNorm before evaluation

LeWM, LeJEPA and LeVJEPA projectors use BatchNorm. Their running averages lag behind features that move during training, which makes eval-mode embeddings drift from train-mode ones. Before evaluating, re-estimate the statistics as exact averages:

```python
import itertools

from torch_world.utils import recompute_batchnorm

recompute_batchnorm(model, itertools.islice(loader, 50), lambda b: model(b["pixels"], b["actions"]))
model.eval()
```

## Frozen encoders

DINO-WM, RoboJEPA and VL-JEPA keep their pre-trained encoder frozen and in eval mode, even after `model.train()`. Pass `model.parameters()` to the optimizer as usual: frozen parameters have no gradient. For VL-JEPA, use `model.param_groups(lr)` to train the Y-encoder at a lower learning rate.

When the encoder is expensive, encode the dataset once and train on the cached features, as [`MedicalVQA.encode`](../datasets/medical-vqa.md#cached-encoder-features) does.

## Multiple GPUs

Models are plain `nn.Module`s, so `DistributedDataParallel` and FSDP2's `fully_shard` wrap them directly. For video-scale data, encode clips once into shards with `write_latent_shard` and stream them with `LatentClips`, which memory-maps each shard so the dataset can be larger than RAM:

```python
from torch.utils.data import DataLoader, DistributedSampler
from torch_world.datasets import LatentClips

data = LatentClips("runs/dz/data")          # every *.pt shard: {"latents": (N, ...), "actions": (N, ...), ...}
loader = DataLoader(data, batch_size=32, sampler=DistributedSampler(data), num_workers=4)
```

`torch_world.benchmarks.dreamzero` is a complete multi-GPU, multi-node pipeline for DreamZero: distributed encoding, DDP or FSDP2 training, resumable checkpoints and distributed evaluation. See [Training at scale](../models/wam/dreamzero.md#training-at-scale).
