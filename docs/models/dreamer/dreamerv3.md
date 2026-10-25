# DreamerV3

> Hafner et al., *Mastering Diverse Domains through World Models*, 2023. [Paper](https://arxiv.org/abs/2301.04104) · [Code](https://github.com/danijar/dreamerv3) · [Site](https://danijar.com/dreamerv3) · [Notebook](../../examples/action/dreamerv3.ipynb)

DreamerV3's world model is a recurrent state-space model (RSSM). The model state $s_t = (h_t, z_t)$ joins a deterministic GRU state and a stochastic state of `stoch` categoricals with `classes` classes each:

$$
h_t = f(h_{t-1}, z_{t-1}, a_{t-1}), \qquad z_t \sim q(z_t \mid h_t, x_t), \qquad \hat z_t \sim p(\hat z_t \mid h_t)
$$

The posterior $q$ sees the frame $x_t$; the prior $p$ predicts the latent from $h_t$ alone and is what imagination uses. Both categoricals mix 1% uniform into their probabilities (unimix) and are sampled with straight-through gradients. From $s_t$ a CNN decoder reconstructs the frame, a reward head and a continue head predict $r_t$ and $c_t$. KL balancing with free bits trains the prior toward the posterior strongly and the posterior toward the prior weakly:

$$
\mathcal{L} = \lVert \hat x_t - x_t \rVert^2 - \ln p(r_t \mid s_t) - \ln p(c_t \mid s_t)
+ \beta_\text{dyn} \max\big(1, \operatorname{KL}[\operatorname{sg}(q) \,\Vert\, p]\big)
+ \beta_\text{rep} \max\big(1, \operatorname{KL}[q \,\Vert\, \operatorname{sg}(p)]\big)
$$

with $\beta_\text{dyn} = 0.5$, $\beta_\text{rep} = 0.1$ and KLs summed over the categoricals. Rewards (and values) are regressed as two-hot targets over 255 bins evenly spaced in symlog space on $[-20, 20]$; the reward head and the critic start with zero output weights.

**Actor-critic in imagination.** From every posterior state of the batch, the actor and the prior imagine `horizon = 15` steps. The critic regresses the λ-returns $R_t = r_{t+1} + \gamma c_{t+1} \big((1 - \lambda) v_{t+1} + \lambda R_{t+1}\big)$, $R_H = v_H$ ($\gamma = 0.997$, $\lambda = 0.95$), computed with the current critic, plus the outputs of its own EMA (decay 0.98) as a regularizer. The actor maximizes $(R_t - v_t) / \max(1, S)$, with $S$ the EMA (0.99) of the 5th-to-95th percentile range of the returns, plus $\eta = 3 \cdot 10^{-4}$ times its entropy. All terms are weighted by the cumulative imagined discount $\prod \gamma c$.

## Usage

```python
from torch_world.models import DreamerV3
from torch_world.planning import CEM

model = DreamerV3(action_dim=2)            # 64x64 RGB; small defaults, see below for the paper's sizes
opt = torch.optim.Adam(model.param_groups())   # world model 1e-4, actor and critic 3e-5

out = model(obs, actions)                  # reward-free: (B, T + 1, 3, 64, 64), (B, T, A)
# out: {"loss", "image_loss", "dyn_loss", "rep_loss", "kl"}
out = model(obs, actions, rewards)         # + rewards (B, T) [and terminated (B, T)]
# out adds: "reward_loss", "continue_loss", "critic_loss", "actor_loss", "entropy"
opt.zero_grad()
out["loss"].backward()
model.clip_grad_norm()                     # 1000 for the world model, 100 for the actor and the critic
opt.step()
model.update_target()                      # EMA of the critic, decay 0.98

z = model.encode(obs)                      # filtered model states (B, T + 1, deter_dim + stoch * classes)
frames = model.decode(model.rollout(z[:, :1], plan))   # imagined frames (B, H + 1, 3, 64, 64)
plan = CEM(horizon=8, action_dim=2)(model, model.encode(obs[:, :1]), goal_state)   # plan on the model state
action = model.act(state)                  # the actor's mean action; state = model.step(state, action, next_obs)
```

`forward` sums the world-model, critic and actor losses. Their gradients reach disjoint parameters: imagination runs through the world model with detached weights, so the actor's gradient flows through the dynamics into the actions but never updates the model, and the critic trains on detached states. One backward pass thus matches the reference's three optimizers. Without `rewards`, only the world model trains.

`encode`, `predict` and `rollout` work on the flattened model state `(..., deter_dim + stoch * classes)`, `history = 1`. `encode` filters a clip from the learned initial state; a single frame gives the state after one posterior step. By default the stochastic part is the mode of each categorical, so encodings and rollouts are deterministic; `sample=True` draws it instead, as in training. `step(state, action, obs)` advances the filter online, for running the actor in an environment.

## Choices and deviations

- **Version.** The hyperparameters are those of the paper's first version (Table W.1), as in the 2023 reference code: $\beta_\text{dyn} = 0.5$, horizon 15, actor gradients backpropagated through the dynamics for continuous actions (`actor_grad="dynamics"`). The 2024 revision uses $\beta_\text{dyn} = 1$, horizon 16, Reinforce for every action space (`actor_grad="reinforce"`, `dyn_coef=1.0`, `horizon=16`) and adds a replay-buffer value loss (scale 0.3), not included here.
- **Actor.** Continuous actions only: a Gaussian with mean $\tanh(\mu)$ and std in $[0.1, 1]$, as in the reference. The baseline $v_t$ is subtracted as a constant. `act` clips to $[-1, 1]$.
- **Networks.** LayerNorm and SiLU throughout; the GRU is the reference's single LayerNorm-ed linear layer with an update-gate bias of $-1$. The encoder is stride-2 convolutions of kernel 4 with doubling channels down to 4×4, the decoder its transpose; images are centred by $-0.5$ and reconstructed with a squared error summed over pixels. The initial state is learned, and the first step of a clip takes a zero action from it; actions entering the RSSM are scaled to at most 1 in magnitude, and imagined continue flags are the continue head's mode, all as in the reference.
- **Sizes.** Defaults are small, for notebooks: `deter_dim=256`, `stoch=32`, `classes=16`, `hidden_dim=256`, `mlp_layers=2`, `cnn_depth=16`. The paper's presets (first version, Table B.1), all with 32 categoricals of 32 classes:

| preset | GRU units (`deter_dim`) | CNN multiplier (`cnn_depth`) | dense units (`hidden_dim`) | MLP layers | parameters |
|---|---|---|---|---|---|
| XS | 256 | 24 | 256 | 1 | 8M |
| S | 512 | 32 | 512 | 2 | 18M |
| M | 1024 | 48 | 640 | 3 | 37M |
| L | 2048 | 64 | 768 | 4 | 77M |
| XL | 4096 | 96 | 1024 | 5 | 200M |

  The 2024 revision trains six sizes from 12M to 400M parameters.
- **Optimizer.** `param_groups` gives Adam with the paper's rates and epsilons; the reference's later switch to LaProp with adaptive gradient clipping is not included.
- **Goal planning.** The paper acts with the actor only. Planning with `CEM` toward an encoded goal image is an addition of this library. `cost` is the squared distance between the final imagined model state and the goal's, on the whole state by default (`cost_on="full"`); `cost_on="stoch"` or `"deter"` compares only the categorical or only the recurrent part. A goal image encoded on its own gets the recurrent state of a first step, which does not depend on the image, so only the stochastic part locates the goal; still, in a 4-epoch tuning run on Two-Rooms the full state planned as well as or better than the stochastic part (same room 100% vs 94%, other room 25% vs 12%, 32 tasks each), and the recurrent part alone was worst (50%, 25%).

## Reference

Arguments and methods: [`DreamerV3`](../../api/models.md#torch_world.models.DreamerV3) in the API reference.
