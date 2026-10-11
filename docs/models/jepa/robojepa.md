# RoboJEPA

> Zholus et al., *RoboJEPA: Scaling Robotic Latent World Models*, 2026. [Paper](https://arxiv.org/abs/2610.10515) · [Code](https://github.com/facebookresearch/robo_jepa) · [Notebook](../../examples/action/robojepa.ipynb)

RoboJEPA trains one action-conditioned predictor in the latent space of a frozen V-JEPA 2.1 encoder, on data from many robot embodiments with several cameras each.

- Every timestep contributes the patch tokens of all camera views, one action token and one proprioceptive state token.
- Actions and states are projected with separate weights per embodiment (`EmbodimentLinear`); an embodiment embedding is added to every token.
- Attention is causal over frames within a `window` of past frames; positions use 3D RoPE `(t, h, w)`.
- The loss is L1 on layer-normalized features, under teacher forcing and over `rollout_steps` autoregressive steps, as in V-JEPA 2-AC.

Latents are `(B, T, V, H, W, embed_dim)`: `V` views on an `H × W` patch grid.

## Usage

```python
from torch_world.models import RoboJEPA

model = RoboJEPA(
    embed_dim=192, action_dim=2, state_dim=2, num_views=2, num_embodiments=2,
    predictor_dim=256, depth=6, num_heads=4, window=4, rollout_steps=2, encoder=encoder,
)

z = model.encode(frames, grid=(4, 4))               # frames (B, T, V, C, H, W)
out = model(z, actions, states, embodiment)         # {"loss", "tf_loss", "rollout_loss"}

# one planner, many robots: the embodiment id changes how actions are read
actions = planner(model, z[:, :1], goal, states=states[:, :1], embodiment=ids)
```

`forward` takes encoder features, so they can be precomputed. During rollouts, states past the given ones repeat the last known state. `cost` is L1.

## Reference

Arguments and methods: [`RoboJEPA`](../../api/models.md#torch_world.models.RoboJEPA), [`EmbodimentLinear`](../../api/models.md#torch_world.models.jepa.EmbodimentLinear) in the API reference.
