# LeJEPA

> Balestriero & LeCun, *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*, 2025. [Paper](https://arxiv.org/abs/2511.08544) · [Code](https://github.com/galilai-group/lejepa) · [Notebook](../../examples/vision/lejepa.ipynb)

LeJEPA has no teacher network, stop-gradient or schedule. Every view's projection predicts the mean projection of the global views, and **SIGReg** (Sketched Isotropic Gaussian Regularization) pushes the projections of every view toward an isotropic Gaussian, which rules out collapse:

$$
\mathcal{L} = (1 - \lambda)\,\big\lVert \bar z_{\text{global}} - z_v \big\rVert^2 + \lambda\,\operatorname{SIGReg}(z)
$$

SIGReg projects the embeddings on random unit directions and measures each projection's distance from a standard normal with the Epps-Pulley statistic.

## Usage

```python
from torch import nn

from torch_world.models import LeJEPA
from torch_world.nn import MLP

model = LeJEPA(backbone, MLP([192, 512, 512, 64], norm=nn.BatchNorm1d), lamb=0.05)

views = [global_crop(x) for _ in range(2)] + [local_crop(x) for _ in range(4)]
out = model(views, num_global=2)        # {"loss", "pred_loss", "sigreg_loss"}
```

`views` is a list of `(B, ...)` tensors (sizes may differ between global and local crops) or one `(B, V, ...)` tensor. The backbone maps a view to an embedding; the projector is usually an MLP with BatchNorm, so call [`recompute_batchnorm`](../../guides/training.md#batchnorm-before-evaluation) before evaluating.

## Reference

Arguments and methods: [`LeJEPA`](../../api/models.md#torch_world.models.LeJEPA) in the API reference.
