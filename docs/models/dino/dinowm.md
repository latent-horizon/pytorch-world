# DINO-WM

> Zhou et al., *DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning*, ICML 2025. [Paper](https://arxiv.org/abs/2411.04983) · [Code](https://github.com/gaoyuezhou/dino_wm) · [Notebook](../../examples/action/dinowm.ipynb)

DINO-WM learns dynamics over the patch features of a frozen pre-trained encoder (DINOv2). Proprioception and action embeddings are tiled over every patch and concatenated on the feature dim. A ViT predictor with frame-level causal attention predicts the next frame's visual and proprioceptive features, trained with MSE against detached targets.

Latents from `encode` are `(B, T, P, D + proprio_emb_dim)`: `P` patches per frame.

## Usage

```python
from torch_world.models import DINOWM
from torch_world.nn import PatchTokens

encoder = PatchTokens.from_hub("dinov2_vits14")   # frozen, 224 px, ImageNet-normalized: 256 tokens of 384
model = DINOWM(encoder, num_patches=256, embed_dim=384, action_dim=2, proprio_dim=4, history=3)

out = model(pixels, actions, proprio)    # {"loss", "visual_loss", "proprio_loss"}

# planning: proprioception is forwarded to rollout
z = model.encode(obs[:, None], proprio[:, None])
goal = model.encode(goal_pixels[:, None], goal_proprio[:, None])[:, 0]
actions = planner(model, z, goal)
```

[`PatchTokens`](../../api/nn.md#torch_world.nn.PatchTokens) wraps a DINOv2 model to return its normalized patch tokens; it takes float images in `[0, 1]` or uint8 images and resizes them. Any frozen module mapping `(N, C, H, W)` to `(N, num_patches, embed_dim)` works: the notebooks use [`PixelPatches`](../../api/nn.md#torch_world.nn.PixelPatches), raw pixel patches, to stay small. The encoder stays in eval mode even after `model.train()`.

## Reference

Arguments and methods: [`DINOWM`](../../api/models.md#torch_world.models.DINOWM), [`PatchTokens`](../../api/nn.md#torch_world.nn.PatchTokens) in the API reference.
