# TD-MPC2

> Hansen et al., *TD-MPC2: Scalable, Robust World Models for Continuous Control*, ICLR 2024. [Paper](https://arxiv.org/abs/2310.16828) · [Code](https://github.com/nicklashansen/tdmpc2) · [Site](https://www.tdmpc2.com) · [Notebook](../../examples/action/tdmpc2.ipynb)

TD-MPC2 learns five components jointly, all MLPs with LayerNorm and Mish:

- an **encoder** to latents normalized by SimNorm: a softmax over each group of 8 features;
- **latent dynamics** `z' = d(z, a)`, also ending in SimNorm;
- a **reward** head and an ensemble of `num_q` **Q-functions**, which predict distributions over 101 bins evenly spaced in symlog space;
- a tanh-Gaussian **policy prior** `π(z)`.

From the first observation of a window, the dynamics unroll the window's actions. Each imagined latent is pulled toward the encoding of the real next observation, and predicts the reward and the TD target, with soft cross-entropy against two-hot targets. Step `t` is weighted by `ρ^t`:

$$
\mathcal{L} = \frac{1}{H}\sum_{t=0}^{H-1} \rho^t \Big( c_1 \lVert \hat z_{t+1} - \operatorname{sg}(h(s_{t+1})) \rVert^2 + c_2\, \operatorname{CE}(\hat r_t, r_t) + c_3\, \operatorname{CE}(\hat q_t, r_t + \gamma\, \bar Q(z_{t+1}, \pi(z_{t+1}))) \Big)
$$

The target `Q̄` is the minimum of two target Q-functions drawn from the ensemble, updated by Polyak averaging. The policy prior maximizes `Q(z, π(z))`, scaled by a running estimate of its 5th to 95th percentile range, plus an entropy bonus, on detached latents and without updating the Q-functions.

## Usage

```python
from torch_world.models import TDMPC2
from torch_world.models.tdmpc import state_encoder
from torch_world.planning import MPPI

model = TDMPC2(state_encoder(obs_dim=4, latent_dim=64, hidden_dim=128), action_dim=2, latent_dim=64, mlp_dim=128,
               discount=TDMPC2.episode_discount(25))
opt = torch.optim.Adam(model.param_groups(3e-4))   # encoder at 0.3 lr, policy with eps=1e-5

out = model(obs, actions, rewards)    # (B, H + 1, obs_dim), (B, H, A), (B, H)
# out: {"loss", "consistency_loss", "reward_loss", "value_loss", "pi_loss"}
opt.zero_grad()
out["loss"].backward()
model.clip_grad_norm(20.0)            # the world model and the policy, separately
opt.step()
model.update_target(0.99)             # Polyak update of the target Q-functions

planner = MPPI(horizon=3, action_dim=2)
plan = planner(model, model.encode(obs[:, None]), explore=True)   # (B, 3, A); execute plan[:, 0]
planner.reset()                       # between episodes
```

Actions are in `[-1, 1]`. `encoder` is any module mapping observations `(N, ...)` to SimNorm latents `(N, latent_dim)`. `state_encoder` builds the reference's MLP for state vectors, and `pixel_encoder` its convolutional encoder for 64×64 images, with random-shift augmentation. Its latent has 512 features, the default `latent_dim`. The defaults are the paper's 5M-parameter model and its loss coefficients.

`forward` returns the world-model loss and the policy loss summed. Their gradients reach disjoint parameters, so one backward pass gives the same gradients as the reference's two separate updates, except that the policy loss sees the Q-functions from before this step's update, not after. For episodic tasks, pass `episodic=True` and a `terminated` mask `(B, H)`. A termination head then cuts TD targets and imagined returns. The reference's multi-task embeddings are not included.

Planning: `estimate_value(z, actions)` is the discounted sum of predicted rewards plus the mean of two Q-values at the last imagined latent. [`MPPI`](../../guides/planning.md#mppi-planning-for-reward) scores sequences with it.

## Reference

Arguments and methods: [`TDMPC2`](../../api/models.md#torch_world.models.TDMPC2), [`MPPI`](../../api/planning.md#torch_world.planning.MPPI) in the API reference.
