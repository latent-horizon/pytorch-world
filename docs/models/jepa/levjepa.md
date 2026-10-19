# LeVJEPA

> Kuhn et al., *LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics*, 2026. [Paper](https://arxiv.org/abs/2608.27395) · [Code](https://github.com/galilai-group/levjepa) · [Project page](https://levjepa.github.io/) · [Notebook](../../examples/vision/levjepa.ipynb)

LeVJEPA trains a video encoder with the [LeJEPA](lejepa.md) objective. It has no EMA target encoder, no stop-gradient and no predictor network. Each clip gives one **global view** at full resolution and `V` **local views**, spatial crops over the same frames. A shared encoder reads every view out at its `[cls]` token, a projector maps it to `z_v`, and every view predicts the global view while **SIGReg** keeps the projections an isotropic Gaussian:

$$
\mathcal{L} = \frac{1}{V + 1} \sum_{v=0}^{V} \big\lVert z_{\text{global}} - z_v \big\rVert^2 + \lambda\,\operatorname{SIGReg}(z), \qquad \lambda = 0.02
$$

Gradients flow through both sides of the prediction term, because the target comes from the same encoder in the same pass.

`LeVJEPAEncoder` is the video ViT of the paper. It is a [`VisionTransformer`](../../api/nn.md#torch_world.nn.VisionTransformer) with three changes:

- **Token dropping.** In training, a fraction `token_drop` of patch tokens (0.95 in the paper) is dropped uniformly at random after patch embedding, and only the kept tokens enter the transformer. In evaluation every token is kept. Dropping makes training several times cheaper, and in the paper it also raises probe accuracy.
- **Per-frame tokens.** `tubelet_size=1` gives one token per frame and patch. The paper finds this beats temporal patches at a matched token budget.
- **Block-causal attention.** Patches attend bidirectionally within a frame and causally across frames. `[cls]` attends every token and no patch attends `[cls]`, so nothing from a later frame reaches an earlier one, and a stream can be encoded frame by frame. Set `block_causal=False` for full attention.

Positions are 3D RoPE computed from each input's own grid, so global and local views of different sizes share the encoder.

## Usage

```python
from torch import nn

from torch_world.models import LeVJEPA, LeVJEPAEncoder
from torch_world.nn import MLP

encoder = LeVJEPAEncoder(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
                         num_frames=16, tubelet_size=1, token_drop=0.95)
model = LeVJEPA(encoder, MLP([768, 2048, 256], norm=nn.BatchNorm1d), lamb=0.02)

global_view = crop(videos, scale=(0.8, 1.0), size=224)                  # (B, C, T, 224, 224)
local_views = [crop(videos, scale=(0.02, 0.4), size=96) for _ in range(10)]  # each (B, C, T, 96, 96)
out = model(global_view, local_views)   # {"loss", "pred_loss", "sigreg_loss"}

embedding = model.encode(videos)        # (B, 768) [cls] embedding, for probes
```

`local_views` is a list of `(B, C, T, h, w)` tensors or one `(B, V, C, T, h, w)` tensor. Each crop is the same region of every frame. Jitter the color of every view on its own: otherwise every crop of a clip shares its colors, and matching views by color solves the objective without learning anything else. The reference recipe also adds horizontal flips, trains with AdamW (lr `4e-4`, weight decay `0.04`, betas `(0.9, 0.95)`) and keeps an EMA copy of the encoder for evaluation only. The projector has BatchNorm, so call [`recompute_batchnorm`](../../guides/training.md#batchnorm-before-evaluation) before evaluating it. Frozen-feature evaluation uses `model.encode` or the encoder's tokens with an [attentive probe](../../guides/metrics.md#probing-frozen-features).

## Reference

Arguments and methods: [`LeVJEPA`](../../api/models.md#torch_world.models.LeVJEPA) and [`LeVJEPAEncoder`](../../api/models.md#torch_world.models.LeVJEPAEncoder) in the API reference.
