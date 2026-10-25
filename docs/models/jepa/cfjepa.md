# CF-JEPA

> Byrd, Wright and Ha, *CF-JEPA: Improving Robustness of JEPA World Models via Controllability Factorization*, 2026. [Paper](https://arxiv.org/abs/2610.00727) · Code: not released · [Notebook](../../examples/action/cfjepa.ipynb)

CF-JEPA (controllability factorization) is [LeWM](lewm.md) with its latent split in two, $z = [z^c, z^u]$. The paper's setting is an exogenous block MDP: actions move only part of the state, and distractors move on their own. An action-conditioned causal transformer $g$ predicts the next $z^c$ from $z^c$ alone. A smaller action-free causal transformer $h$ predicts the next $z^u$ from $z^u$ alone. Two small MLPs try to recover the action from consecutive latents. The inverse model $\psi$ reads $z^c$ and is trained normally, so $z^c$ must hold what the agent controls. The adversary $\phi$ reads $z^u$ behind a gradient reversal layer, so the encoder learns to strip action information out of $z^u$. SIGReg, as in LeWM, keeps the latent from collapsing. Planning compares only $z^c$ with the goal, so a disc drifting through the goal image does not count as distance.

$$
\mathcal{L} = \big\lVert [g(z^c_t, a_t),\, h(z^u_t)] - z_{t+1} \big\rVert^2
+ \alpha \big\lVert \psi(z^c_t, z^c_{t+1}) - a_t \big\rVert^2
+ \beta \big\lVert \phi(\operatorname{GRL}(z^u_t, z^u_{t+1})) - a_t \big\rVert^2
+ \lambda\,\operatorname{SIGReg}(z)
$$

## Usage

```python
from torch_world.models import CFJEPA
from torch_world.nn import CLSEncoder

model = CFJEPA(CLSEncoder(img_size=64, patch_size=8, embed_dim=128, depth=4, num_heads=4), action_dim=2,
               embed_dim=128, ctrl_dim=96, history=3, depth=4, num_heads=4, head_dim=32, mlp_dim=512, proj_dim=512)

out = model(pixels, actions)              # (B, history + 1, C, H, W), (B, history + 1, A)
# out: {"loss", "pred_loss", "inverse_loss", "adversary_loss", "sigreg_loss"}

z = model.encode(pixels)                  # (B, T, embed_dim), controllable dims first
zc, zu = model.split(z)                   # (B, T, ctrl_dim), (B, T, embed_dim - ctrl_dim)
cost = model.cost(model.rollout(z[:, :1], plan), goal)  # distance on z^c only
```

The paper uses $\alpha = 3$ on Two-Room and Push-T and $\alpha = 10$ on Reacher and Cube, with $\beta = 1$ and $\lambda = 0.09$ throughout. These are the defaults, with `alpha = 3`. `rollout` advances both parts, and [`CEM`](../../api/planning.md#torch_world.planning.CEM) uses the model's `cost`, which reads only $z^c$, so the planners work unchanged. The adversary's loss should stay high during training: it measures how much action information is left in $z^u$.

The paper does not give every detail, so these are this implementation's choices:

- The encoder output goes through LeWM's projector before the split. The paper mentions no projector, but it builds on the LeWM code, and keeping the projector makes CF-JEPA match the LeWM baseline.
- SIGReg runs on the whole latent. The paper says only that it uses LeWM's SIGReg.
- $h$ has half of $g$'s depth. The paper says only that $h$ is smaller.
- $\psi$ and $\phi$ are two-layer GELU MLPs with `inv_dim` hidden units. The paper gives the depth and activation but not the width.
- The gradient reversal scale is `reversal = 1`. The paper does not state it.
- Both predictors use the same `history` window as LeWM.

The paper splits ViT-tiny's 192 dimensions into 128 for $z^c$ and 64 for $z^u$. The notebook uses a 128-dimensional latent split 96/32. The projectors use BatchNorm: call [`recompute_batchnorm`](../../guides/training.md#batchnorm-before-evaluation) before planning.

## Reference

Arguments and methods: [`CFJEPA`](../../api/models.md#torch_world.models.CFJEPA) in the API reference.
