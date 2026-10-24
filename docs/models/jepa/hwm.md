# HWM

> Zhang et al., *Hierarchical Planning with Latent World Models*, 2026. [Paper](https://arxiv.org/abs/2604.03208) · [Code](https://github.com/kevinghst/HWM_PLDM) · [Notebook](../../examples/action/hwm.ipynb)

HWM plans over long horizons with two action-conditioned world models that share one latent space. The low level predicts the next latent from a primitive action. The high level predicts the latent `stride` steps later, the next *waypoint*, from a latent macro-action $l_k = A_\psi(a_{t_k:t_{k+1}-1})$ that encodes the actions in between. Here the low level is a [`LeWM`](lewm.md), so the encoder learns from pixels with SIGReg. The high level reads the low level's latents through a stop-gradient, as HWM trains it on a frozen encoder. Each level has HWM's ℓ1 teacher-forcing loss. The low level also has the ℓ1 multi-step rollout loss $\sum_j \lVert F^{(1)}(z_1, a_{1:j}) - z_{j+1} \rVert_1$:

$$
\mathcal{L} = \mathcal{L}^{(1)}_\text{tf} + \gamma\,\mathcal{L}^{(1)}_\text{roll} + \lambda\,\operatorname{SIGReg}(z) + \mathcal{L}^{(2)}_\text{tf}
$$

Planning is top-down, and the plan is redone after every executed step:
1. The high level searches macro-actions $l_{1:H}$ that minimize $\lVert F^{(2)}(z_1, l_{1:H}) - z_g \rVert_1$.
2. Its first predicted waypoint $\tilde z_1$ becomes the subgoal.
3. The low level searches the next `stride` actions that minimize $\lVert F^{(1)}(z_1, a_{1:h}) - \tilde z_1 \rVert_1$.

## Usage

```python
from torch_world.models import HWM, LeWM
from torch_world.nn import CLSEncoder
from torch_world.planning import CEM, HierarchicalPlanner

low = LeWM(CLSEncoder(img_size=64, patch_size=8, embed_dim=128, depth=4, num_heads=4), action_dim=2, embed_dim=128,
           history=3, depth=4, num_heads=4, head_dim=32, mlp_dim=512, proj_dim=512, lamb=0.05)
model = HWM(low, action_dim=2, stride=4, macro_dim=8, embed_dim=128, history=3, depth=4, num_heads=4, head_dim=32,
            mlp_dim=512)

out = model(pixels, actions)        # (B, 3 * 4 + 1, C, H, W), (B, 3 * 4 + 1, A): 4 waypoints
# out: {"loss", "low_loss", "rollout_loss", "sigreg_loss", "high_loss"}

planner = HierarchicalPlanner(
    CEM(horizon=5, action_dim=8, bounds=(-2, 2)),      # macro-actions toward the goal
    CEM(horizon=4, action_dim=2, bounds=(-1, 1)),      # primitive actions toward the first waypoint
)
plan = planner(model, model.encode(obs[:, None]), model.encode(goal_images[:, None])[:, 0])  # (B, 4, 2)
```

The macro-action encoder ends with a BatchNorm without affine parameters, so training macro-actions are standardized. The high-level planner can then sample them from $\mathcal N(0, I)$ and clip them to ±2. The paper initializes from, and clips to, the 2–98% quantiles of training macro-actions. With `shrink_every`, [`HierarchicalPlanner`](../../api/planning.md#torch_world.planning.HierarchicalPlanner) shortens the high horizon as the episode runs, as the paper does.

The paper differs from this implementation in a few ways:
- **Encoder:** frozen pre-trained encoders (V-JEPA 2, DINOv2) or PLDM's, where here a LeWM encoder learns from pixels.
- **Waypoints:** variable spacing with a transformer macro-action encoder on Franka and Push-T. The fixed stride and MLP encoder used here are the paper's maze setup.
- **Planner:** MPPI in the maze, where this implementation runs CEM at both levels.

Call [`recompute_batchnorm`](../../guides/training.md#batchnorm-before-evaluation) before planning, as for LeWM.

## Reference

Arguments and methods: [`HWM`](../../api/models.md#torch_world.models.HWM) and [`HierarchicalPlanner`](../../api/planning.md#torch_world.planning.HierarchicalPlanner) in the API reference.
