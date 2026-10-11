# I-JEPA

> Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*, CVPR 2023. [Paper](https://arxiv.org/abs/2301.08243) · [Code](https://github.com/facebookresearch/ijepa) · [Notebook](../../examples/vision/ijepa.ipynb)

I-JEPA encodes one large visible context block of an image and predicts the features of several target blocks. Targets are layer-normalized features of an EMA copy of the encoder, so the model never reconstructs pixels. The loss is smooth L1.

$$
\mathcal{L} = \operatorname{smoothL1}\big(g_\phi(f_\theta(x_{\text{ctx}}), m_{\text{tgt}}),\ \operatorname{LN}(\bar f_{\bar\theta}(x))_{m_{\text{tgt}}}\big)
$$

## Usage

```python
from torch_world.masks import MultiBlockMask
from torch_world.models import IJEPA
from torch_world.nn import MaskedPredictor, VisionTransformer
from torch_world.utils import linear_schedule

encoder = VisionTransformer(img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=6)
predictor = MaskedPredictor(encoder.grid, 192, predictor_dim=96, depth=4, num_heads=6)
model = IJEPA(encoder, predictor)
masker = MultiBlockMask(encoder.grid, min_keep=4)

for step in range(steps):
    masks_enc, masks_pred = masker(len(images))
    loss = model(images, masks_enc, masks_pred)["loss"]
    opt.zero_grad()
    loss.backward()
    opt.step()
    model.update_target(linear_schedule(0.996, 1.0, step, steps))
```

At paper scale: `VisionTransformer.from_size("huge", img_size=224, patch_size=14)` with a 12-layer, 384-wide predictor. After training, use `model.target_encoder` as the feature extractor.

## Reference

Arguments and methods: [`IJEPA`](../../api/models.md#torch_world.models.IJEPA) in the API reference.
