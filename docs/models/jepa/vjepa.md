# V-JEPA

> Bardes et al., *Revisiting Feature Prediction for Learning Visual Representations from Video*, 2024. [Paper](https://arxiv.org/abs/2404.08471) · [Code](https://github.com/facebookresearch/jepa)
>
> Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, 2025. [Paper](https://arxiv.org/abs/2506.09985) · [Code](https://github.com/facebookresearch/vjepa2) · [Notebook](../../examples/vision/vjepa.ipynb)

V-JEPA applies feature prediction to video. Masks are tubes that span time, so a target patch cannot be copied from a neighboring frame. Several masking strategies run together (short-range and long-range in the paper), and strategy `i` uses the predictor's mask token `i`. The loss is

$$
\mathcal{L} = \frac{1}{p}\,\operatorname{mean}\,\lvert z - h \rvert^{p} \;+\; \lambda\,\operatorname{mean}\,\operatorname{ReLU}\big(1 - \sqrt{\operatorname{Var}(z) + \epsilon}\big)
$$

averaged over strategies, with `p = loss_exp` and `λ = reg_coeff`. **V-JEPA 2** keeps this objective with 3D RoPE: build the encoder and predictor with `use_rope=True`.

## Usage

```python
from torch_world.masks import MultiBlock3DMask
from torch_world.models import VJEPA
from torch_world.nn import MaskedPredictor, VisionTransformer

encoder = VisionTransformer(img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=6,
                            num_frames=8, tubelet_size=2, use_rope=True)
predictor = MaskedPredictor(encoder.grid, 192, 96, depth=4, num_heads=6,
                            num_mask_tokens=2, zero_init_mask_tokens=True, use_rope=True)
model = VJEPA(encoder, predictor)
maskers = [
    MultiBlock3DMask(encoder.grid, spatial_scale=(0.15, 0.15), num_blocks=8),
    MultiBlock3DMask(encoder.grid, spatial_scale=(0.7, 0.7), num_blocks=2),
]

videos = episodes["pixels"].transpose(1, 2)            # (B, C, T, H, W)
masks = [m(len(videos)) for m in maskers]
out = model(videos, [enc for (enc,), _ in masks], [pred for _, (pred,) in masks])
# out: {"loss", "jepa_loss", "reg_loss"}
```

Call `model.update_target(momentum)` after every optimizer step.

## Reference

Arguments and methods: [`VJEPA`](../../api/models.md#torch_world.models.VJEPA) in the API reference.
