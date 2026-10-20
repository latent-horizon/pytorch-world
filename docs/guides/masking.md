# Masking

Masks decide what a JEPA sees and what it has to predict. In `torch_world` a mask is a `(B, K)` tensor of token indices to keep, and a masking strategy is a callable that returns lists of them.

## Image blocks: `MultiBlockMask`

The I-JEPA strategy on a 2D patch grid. Each image gets `num_pred` target blocks (15-20% of the grid each) and `num_enc` large context blocks (85-100%) that avoid the targets.

```python
from torch_world.masks import MultiBlockMask
from torch_world.nn import VisionTransformer

encoder = VisionTransformer(img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=6)
masker = MultiBlockMask(encoder.grid, min_keep=4)

masks_enc, masks_pred = masker(batch_size=128)
# masks_enc:  [ (128, K_enc) ]          one context block
# masks_pred: [ (128, K_pred) ] * 4     four target blocks
```

Block sizes are shared across the batch and every mask is cropped to the shortest one, so the masks stack. Pass `allow_overlap=True` to let context and targets overlap, and a `generator` for reproducible masks.

::: warning `min_keep` on small grids
A block must hold more than `min_keep` patches. The default of 10 is meant for 14×14 grids and up; lower it on small grids, or sampling raises `ValueError`.
:::

## Video tubes: `MultiBlock3DMask`

The V-JEPA strategy on a `(T, H, W)` token grid. `num_blocks` spatial blocks extend through time as tubes, so the model cannot copy a target patch from a neighboring frame. Their union is the target and the rest is the context.

```python
from torch_world.masks import MultiBlock3DMask

video_encoder = VisionTransformer(img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=6,
                                  num_frames=8, tubelet_size=2, use_rope=True)
maskers = [
    MultiBlock3DMask(video_encoder.grid, spatial_scale=(0.15, 0.15), num_blocks=8),  # short-range
    MultiBlock3DMask(video_encoder.grid, spatial_scale=(0.7, 0.7), num_blocks=2),    # long-range
]
masks = [m(batch_size=64) for m in maskers]
masks_enc = [enc for (enc,), _ in masks]
masks_pred = [pred for _, (pred,) in masks]
```

[`VJEPA`](../models/jepa/vjepa.md) takes one context/target pair per strategy, and the predictor uses a separate mask token for each (`num_mask_tokens=2`).

## Applying masks

```python
from torch_world.masks import apply_masks

tokens = encoder.patch_embed(images)          # (B, N, D)
context = apply_masks(tokens, masks_enc)      # (len(masks_enc) * B, K, D)
```

`VisionTransformer.forward(x, masks)` does this for you and encodes only the kept tokens, as JEPA context encoders do. [`MaskedPredictor`](../api/nn.md) takes the same lists and predicts every target block for every context block.

## Frame-causal attention

World models predict frame `t + 1` from frames up to `t`. `frame_causal_mask` builds the boolean attention mask in which each token sees every token of its own frame and of the `window` frames before it.

```python
from torch_world.masks import frame_causal_mask

mask = frame_causal_mask(num_frames=4, tokens_per_frame=16, window=None)  # (64, 64), True attends
```

## Visualizing masks

```python
from torch_world import plots

layers = {"context": masks_enc[0][0], **{f"target {i + 1}": m[0] for i, m in enumerate(masks_pred)}}
plots.masks(image, encoder.grid, layers)
```
