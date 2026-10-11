# LeWM

> Maes et al., *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels*, 2026. [Paper](https://arxiv.org/abs/2603.19312) · [Code](https://github.com/lucas-maes/le-wm) · [Notebook](../../examples/action/lewm.ipynb)

LeWorldModel learns its encoder and predictor jointly from raw pixels, with no frozen backbone, EMA target or stop-gradient. Each frame is encoded to one embedding (the encoder's CLS token) and projected. A causal transformer, conditioned on action embeddings through adaLN-zero, predicts the next embedding from the last `history` frames.

$$
\mathcal{L} = \big\lVert \hat z_{t+1} - z_{t+1} \big\rVert^2 + \lambda\,\operatorname{SIGReg}(z)
$$

## Usage

```python
from torch_world.models import LeWM
from torch_world.nn import CLSEncoder

model = LeWM(CLSEncoder(img_size=32, patch_size=8, embed_dim=64), action_dim=2, embed_dim=64, history=3, depth=3, num_heads=4,
             head_dim=16, mlp_dim=256, proj_dim=256, lamb=0.01)

out = model(pixels, actions)        # (B, history + 1, C, H, W), (B, history + 1, A)
# out: {"loss", "pred_loss", "sigreg_loss"}

z = model.encode(pixels)            # (B, T, embed_dim)
```

`encoder` maps images `(N, C, H, W)` to embeddings `(N, embed_dim)`; a ViT-tiny CLS token in the paper, which [`CLSEncoder`](../../api/nn.md#torch_world.nn.CLSEncoder) provides. The paper uses `lamb = 0.09`; on scenes with very few degrees of freedom a weaker prior keeps the latent geometry smoother. When actions are stacked over a frameskip, `action_dim` is the stacked size.

The projectors use BatchNorm: call [`recompute_batchnorm`](../../guides/training.md#batchnorm-before-evaluation) before planning. The [Quick start](../../getting-started/quickstart.md) trains and plans with LeWM end to end.

## Reference

Arguments and methods: [`LeWM`](../../api/models.md#torch_world.models.LeWM) in the API reference.
