# PLDM

> Sobal et al., *Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models*, 2025. [Paper](https://arxiv.org/abs/2502.14819) · [Code](https://github.com/vladisai/PLDM) · [Notebook](../../examples/action/pldm.ipynb)

PLDM learns a latent dynamics model from reward-free trajectories of any quality, then plans with it toward goal images. Each frame is encoded to one latent and normalized by a LayerNorm. A GRU whose hidden state is that latent predicts the next latent from the action, through the same LayerNorm. In training it unrolls open loop from the first frame over the whole window. There is no decoder, EMA target or stop-gradient. Collapse is prevented by VICReg's variance and covariance terms over the batch at every step, a temporal smoothness term and an inverse dynamics model that regresses each action from consecutive latents:

$$
\mathcal{L} = \sum_{t=1}^{H} \big\lVert \hat z_t - z_t \big\rVert^2 + \alpha\,\mathcal{L}_\text{var}(z) + \beta\,\mathcal{L}_\text{cov}(z) + \delta \sum_{t=0}^{H-1} \big\lVert z_{t+1} - z_t \big\rVert^2 + \omega \sum_{t=0}^{H-1} \big\lVert a_t - \operatorname{IDM}(z_t, z_{t+1}) \big\rVert^2
$$

where $\hat z_t$ are the multi-step predictions from $z_0$. Planning minimizes the latent distance to the goal along the imagined path, $\sum_t \lVert \hat z_t - z_g \rVert^2$, and replans every step.

## Usage

```python
from torch_world.models import PLDM, PLDMEncoder
from torch_world.planning import CEM

model = PLDM(PLDMEncoder(img_size=64, latent_dim=128), action_dim=2, latent_dim=128)

out = model(pixels, actions)        # (B, H + 1, C, H, W), (B, H + 1, A)
# out: {"loss", "pred_loss", "var_loss", "cov_loss", "smooth_loss", "idm_loss"}

z = model.encode(pixels)            # (B, T, latent_dim)
planner = CEM(horizon=20, action_dim=2, bounds=(-1, 1), objective="path")
plan = planner(model, model.encode(obs[:, None]), model.encode(goal_images[:, None])[:, 0])
```

`encoder` maps images `(N, C, H, W)` to latents `(N, latent_dim)`; [`PLDMEncoder`](../../api/models.md#torch_world.models.PLDMEncoder) is a small IMPALA-style CNN like the paper's. The default weights `var_coeff = 4.0`, `cov_coeff = 6.9`, `smooth_coeff = 0.75` and `idm_coeff = 1.0` are the reference's Two-Rooms ones, with the covariance normalized by $D(D-1)$ as in its code. The paper plans with MPPI over an ensemble of predictors; CEM with the `"path"` objective minimizes the same running cost. The [notebook](../../examples/action/pldm.ipynb) trains it on [`Macaw(rooms=True)`](../../guides/environments.md#macaw), the Two-Rooms task flown by a macaw and plans through the door.

## Reference

Arguments and methods: [`PLDM`](../../api/models.md#torch_world.models.PLDM) in the API reference.
