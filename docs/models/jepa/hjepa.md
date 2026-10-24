# H-JEPA

> Zhang et al., *H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning*, 2026. [Paper](https://arxiv.org/abs/2610.06805) · [Code](https://github.com/kevinghst/H-JEPA) · [Notebook](../../examples/action/hjepa.ipynb)

H-JEPA stacks JEPA world models and trains them jointly from pixels. Level 1 is a [`LeWM`](lewm.md): a ViT whose CLS token embeds each frame, and a causal transformer that predicts the next embedding from the action. Level 2 has its own encoders and predictor:
- **States:** an MLP encodes every `stride`-th level-1 latent, $z^{(2)}_k = E^{(2)}(z^{(1)}_{k s})$.
- **Macro-actions:** a one-layer transformer pools the `stride` level-1 action embeddings between two level-2 states.
- **Predictor:** a causal transformer predicts the next level-2 latent from the current one and the macro-action.

Each level has a teacher-forced prediction loss and SIGReg on its own latents. The level-2 loss also trains the level-1 encoder and action encoder:

$$
\mathcal{L} = \sum_{\ell=1}^{2} \big\lVert F^{(\ell)}(z^{(\ell)}_{<\tau}, a^{(\ell)}_{<\tau}) - z^{(\ell)}_{\tau+1} \big\rVert^2 + \lambda\,\operatorname{SIGReg}(Z^{(\ell)})
$$

Planning is top-down:
1. Level 2 plans macro-actions toward $E^{(2)}(z_g)$.
2. Level 1 plans the next `stride` actions so that $E^{(2)}$ of its last predicted latent matches level 2's first prediction. The match is measured in level-2 space.
3. Once the level-2 horizon is down to one step, level 1 aims at the goal itself.

The paper plans every level by gradient descent through the predictor.

## Usage

```python
from torch_world.models import HJEPA, LeWM
from torch_world.nn import CLSEncoder
from torch_world.planning import GradientPlanner, HierarchicalPlanner

low = LeWM(CLSEncoder(img_size=64, patch_size=8, embed_dim=128, depth=4, num_heads=4), action_dim=2, embed_dim=128,
           history=3, depth=4, num_heads=4, head_dim=32, mlp_dim=512, proj_dim=512, lamb=0.05)
model = HJEPA(low, stride=4, macro_dim=4, embed_dim=128, encoder_dim=256, history=3, depth=4, num_heads=4,
              head_dim=32, mlp_dim=512)

out = model(pixels, actions)        # (B, 3 * 4 + 1, C, H, W), (B, 3 * 4 + 1, A): 4 level-2 states
# out: {"loss", "low_loss", "high_loss", "low_sigreg_loss", "high_sigreg_loss"}

planner = HierarchicalPlanner(
    GradientPlanner(horizon=5, action_dim=4, bounds=(-2, 2)),   # level-2 macro-actions
    GradientPlanner(horizon=4, action_dim=2, bounds=(-1, 1)),   # level-1 actions toward the subgoal
)
plan = planner(model, model.encode(obs[:, None]), model.encode(goal_images[:, None])[:, 0])  # (B, 4, 2)
```

[`GradientPlanner`](../../api/planning.md#torch_world.planning.GradientPlanner) optimizes sampled candidates with Adam and keeps the cheapest, as the paper's solver does without its early stopping. Macro-actions are standardized by a BatchNorm without affine parameters, where the reference ends its pooling with a LayerNorm. The planner can then sample them from $\mathcal N(0, I)$ and clip them to ±2, close to the paper's clipping to the 2–98% quantiles of training macro-actions.

The paper's setting differs in two ways:
- **Stride:** its levels are 2 steps of 5-action blocks apart, where the default here is a `stride` of 4 primitive actions.
- **Inverse dynamics:** its DROID models add an inverse-dynamics term, which this implementation leaves out.

Call [`recompute_batchnorm`](../../guides/training.md#batchnorm-before-evaluation) before planning, as for LeWM.

## Reference

Arguments and methods: [`HJEPA`](../../api/models.md#torch_world.models.HJEPA), [`HierarchicalPlanner`](../../api/planning.md#torch_world.planning.HierarchicalPlanner) and [`GradientPlanner`](../../api/planning.md#torch_world.planning.GradientPlanner) in the API reference.
