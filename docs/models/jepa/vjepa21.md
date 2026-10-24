# V-JEPA 2.1

> Mur-Labadia et al., *V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning*, 2026. [Paper](https://arxiv.org/abs/2603.14482) · [Code](https://github.com/facebookresearch/vjepa2) · [Notebook](../../examples/vision/vjepa21.ipynb)

V-JEPA 2.1 keeps the [V-JEPA 2](vjepa.md) recipe (EMA target, tube masks, 3D RoPE, L1 loss) and makes the patch tokens useful on their own, for segmentation, depth and tracking. Three things change:

- **Context loss.** The predictor also outputs the visible context tokens, through a separate head, and they are regressed onto the target as well. Each one is weighted by $w = 1/\sqrt{d}$, where $d$ is its distance on the token grid to the nearest masked token.
- **Deep self-supervision.** `VJEPA21Encoder` exposes `levels` evenly spaced blocks, the last one being the final block, and each has its own layer norm. The predictor fuses their concatenation with an MLP and predicts every level. Targets are the EMA encoder's levels, each one layer-normalized.
- **Images.** With `images=True`, images `(B, C, H, W)` get their own 2D patch embedding on the first frame of the grid, instead of being repeated into a static clip. A learned modality embedding is added in the encoder and the predictor.

With `p = loss_exp` and `λ = context_coeff`, the loss is

$$
\mathcal{L} = \frac{1}{p}\,\operatorname{mean}_{\text{masked}}\,\lvert z - h \rvert^{p} \;+\; \lambda\,\frac{1}{p}\,\operatorname{mean}_{\text{context}}\, w\,\lvert z - h \rvert^{p}
$$

averaged over masking strategies. The paper warms `λ` up linearly from 0 and holds it at 0.5 for videos and 0.7 for images. `encoder(x)` returns only the normalized final block, like a V-JEPA encoder, so downstream probes need no change.

## Usage

```python
from torch_world.masks import MultiBlock3DMask
from torch_world.models import VJEPA21, VJEPA21Encoder, VJEPA21Predictor
from torch_world.utils import linear_schedule

encoder = VJEPA21Encoder(img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=6,
                         num_frames=8, tubelet_size=2, use_rope=True, levels=4)
predictor = VJEPA21Predictor(encoder.grid, 192, 96, depth=4, num_heads=6, levels=4,
                             num_mask_tokens=2, zero_init_mask_tokens=True, use_rope=True)
model = VJEPA21(encoder, predictor, context_coeff=0.5)
maskers = [
    MultiBlock3DMask(encoder.grid, spatial_scale=(0.15, 0.15), num_blocks=8),
    MultiBlock3DMask(encoder.grid, spatial_scale=(0.7, 0.7), num_blocks=2),
]

videos = episodes["pixels"].transpose(1, 2)            # (B, C, T, H, W)
masks = [m(len(videos)) for m in maskers]
coeff = linear_schedule(0.0, 0.5, max(0, step - warmup_start), warmup_steps)
out = model(videos, [enc for (enc,), _ in masks], [pred for _, (pred,) in masks], context_coeff=coeff)
# out: {"loss", "pred_loss", "context_loss"}
```

Call `model.update_target(momentum)` after every optimizer step. To train on images as well, build the encoder and predictor with `images=True`. Then call the model on `(B, C, H, W)` images, with masks drawn on a one-frame grid `(1, H, W)`, and add the losses of image and video batches before stepping.

The action-conditioned stage of the paper is a frame-causal predictor on frozen V-JEPA 2.1 features, trained with teacher forcing and a two-step rollout loss. That is [RoboJEPA](robojepa.md) with one embodiment.

## Reference

Arguments and methods: [`VJEPA21`](../../api/models.md#torch_world.models.VJEPA21), [`VJEPA21Encoder`](../../api/models.md#torch_world.models.VJEPA21Encoder) and [`VJEPA21Predictor`](../../api/models.md#torch_world.models.VJEPA21Predictor) in the API reference.
