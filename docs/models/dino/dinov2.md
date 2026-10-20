# DINOv2

> Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, TMLR 2024. [Paper](https://arxiv.org/abs/2304.07193) · [Code](https://github.com/facebookresearch/dinov2) · [Notebook](../../examples/vision/dinov2.ipynb)

DINOv2 trains a ViT student to match an EMA teacher. Each image gives `G = 2` **global crops**, which the teacher reads, and `V` **local crops**; the student reads all of them, with part of the global crops' patches masked. A shared head maps tokens to scores over `K` prototypes, and the teacher's scores become targets after **Sinkhorn-Knopp** centering and sharpening by `teacher_temp`. Three terms make the loss:

$$
\mathcal{L} = \mathcal{L}_{\text{DINO}} + \mathcal{L}_{\text{iBOT}} + 0.1\,\mathcal{L}_{\text{KoLeo}}
$$

- **DINO**: the cross-entropy between the teacher's assignment of each global crop's [cls] token and the student's [cls] of every other crop, global or local, averaged over the `G (G + V) - G` pairs.
- **iBOT**: each masked patch of a global crop predicts the teacher's assignment of the same patch, unmasked. The loss is averaged per image, then over all global crops, masked or not.
- **KoLeo**: on the student's global [cls] tokens, minus the mean log distance to the nearest neighbor within each crop's batch, which spreads them.

There is no gradient through the teacher. The student's scores are sharpened by `student_temp = 0.1`.

`DINOv2Encoder` is the ViT: a [cls] token, learned position embeddings resized bicubically to each crop's grid, and a learned **mask token** that replaces the embedding of each masked patch. `DINOHead` is an MLP to a bottleneck, L2 normalization and weight-normalized prototypes. [`BlockwiseMask`](../../api/masks.md#torch_world.masks.BlockwiseMask) draws the masks as the reference does: half of the global crops are masked, with ratios spread over 10-50%, by BEiT-style rectangles.

## Usage

```python
from torch_world.masks import BlockwiseMask
from torch_world.models import DINOv2, DINOHead, DINOv2Encoder

encoder = DINOv2Encoder(img_size=224, patch_size=14, embed_dim=384, depth=12, num_heads=6)
model = DINOv2(encoder, DINOHead(384, num_prototypes=65536))
masker = BlockwiseMask((16, 16))

global_crops = ...  # (2, B, 3, 224, 224): 32-100% of each image
local_crops = ...   # (8, B, 3, 98, 98): 5-32%
mask = masker(2 * B).view(2, B, -1)
out = model(global_crops, local_crops, mask, teacher_temp=0.07)   # {"loss", "dino_loss", "ibot_loss", "koleo_loss"}
out["loss"].backward()
optimizer.step()
model.update_teacher(momentum)

embedding = model.teacher_encoder(images)[:, 0]   # (B, 384) [cls] embedding of the teacher, for probes
```

The reference schedules: `teacher_temp` rises linearly from 0.04 to 0.07 over the first 30 epochs, the teacher momentum follows a cosine from 0.992 to 1, and the prototypes are frozen for the first epoch. Gradients are clipped to norm 3 for the encoder and the head separately. The released models are the teacher's encoder.

## Differences from the reference

- Centering is Sinkhorn-Knopp, as in the paper and its ViT-L/14 and ViT-g/14 recipes. The reference also supports a moving-average center.
- One head serves DINO and iBOT, as in the reference default; the large recipes use a separate iBOT head.
- Blocks are the library's pre-norm blocks, without LayerScale, and the model runs on one device.

## Reference

Arguments and methods: [`DINOv2`](../../api/models.md#torch_world.models.DINOv2), [`DINOv2Encoder`](../../api/models.md#torch_world.models.DINOv2Encoder) and [`DINOHead`](../../api/models.md#torch_world.models.DINOHead) in the API reference.
