# TD-JEPA

> Bagatella, Pirotta, Touati, Lazaric and Tirinzoni, *TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning*, ICLR 2026. [Paper](https://arxiv.org/abs/2510.00739) · Code: not released · [Notebook](../../examples/action/tdjepa.ipynb)

TD-JEPA learns, from reward-free transitions, a state encoder $\phi$, a task encoder $\psi$ and a family of policies $\pi_z$, one for every task embedding $z$. Its predictor is not a one-step dynamics model. $T_\phi(\phi(s), a, z)$ predicts the discounted sum of future $\psi$ embeddings when $\pi_z$ acts from $(s, a)$: the successor features of $\psi$. That sum has a Bellman equation, so the predictor is trained by temporal-difference latent prediction against EMA target networks. A second predictor $T_\psi$ does the same with the roles of $\phi$ and $\psi$ swapped:

$$
\mathcal{L}_\text{TD}(\phi, T_\phi, \psi) = \tfrac12 \big\lVert T_\phi(\phi(s), a, z) - \bar\psi(s') - \gamma\,\bar T_\phi(\bar\phi(s'), a', z) \big\rVert^2,
\qquad a' \sim \pi(\bar\phi(s'), z)
$$

$$
\mathcal{L} = \mathcal{L}_\text{TD}(\phi, T_\phi, \psi) + \mathcal{L}_\text{TD}(\psi, T_\psi, \phi) + \lambda \big(\mathcal{L}_\text{reg}(\phi) + \mathcal{L}_\text{reg}(\psi)\big) - T_\phi(\phi(s), \hat a, z)^\top z,
\qquad \hat a \sim \pi(\phi(s), z)
$$

Bars are target networks. $\mathcal{L}_\text{reg}(x) = \frac{1}{2B(B-1)}\sum_{i \ne j} (x_i^\top x_j)^2 - \frac1B \sum_i x_i^\top x_i$ pushes $\mathbb{E}[x x^\top]$ toward the identity. Since $T_\phi(\phi(s), a, z)^\top z$ is the Q-value of $\pi_z$ for the reward $r(s) = \psi(s)^\top z$, the last term trains each policy to be greedy for its own task. At test time nothing is trained. Given rewarded samples, $z_r = \mathbb{E}[\psi\psi^\top]^{-1}\mathbb{E}[\psi\, r]$ is a linear regression. Given a goal observation $g$, $z_g = \psi(g)$, optionally whitened by $\mathbb{E}[\psi\psi^\top]^{-1}$ (the regression of a reward that is one at $g$). Then $\pi_{z}$ acts.

## Usage

```python
from torch_world.models import TDJEPA
from torch_world.models.jepa import drq_encoder

model = TDJEPA(drq_encoder(256), drq_encoder(50, hidden_layers=2), action_dim=2, discount=0.98)
opt = torch.optim.Adam(model.param_groups(1e-4))

out = model(pixels, actions)              # transitions (B, 2, C, H, W), (B, 2, A); actions[:, 0] is taken at s
# out: {"loss", "td_phi_loss", "td_psi_loss", "ortho_loss", "actor_loss"}
model.update_target(0.995)                # after each optimizer step

z_goal = model.goal_task(goal_images)     # (B, task_dim), no training
z_reward = model.infer_task(frames, r)    # (task_dim,) from rewards r (N,) at frames (N, C, H, W)
policy = model.controller(z_goal)         # observation (B, C, H, W) -> action (B, A)
action = model.policy(model.encode(obs[:, None])[:, 0], z_goal)
q = model.value(z, action, z_goal)        # T_phi(z, a, task) . task
```

`TDJEPA` subclasses `JEPA`, not `WorldModel`: its predictor forecasts discounted future embeddings under a policy, not the next latent, so there is nothing for `CEM` or `MPPI` to unroll. It acts with its own task-conditioned policy instead. `forward` returns the encoder, predictor and actor losses together. Their gradients reach disjoint parameters: the actor sees detached $\phi$ and a predictor whose parameters it cannot update, and $\bar\psi$, $\bar\phi$ are targets. One backward pass therefore matches Algorithm 1's three updates.

The defaults follow the paper's DMC-from-pixels column (Table 4): $d_\phi = 256$, $d_\psi = 50$, predictor width 1024, $\gamma = 0.98$, $\lambda = 1$, learning rate $10^{-4}$ with Adam, $p_\text{goal} = 0.5$ and a Gaussian actor with fixed std 0.2. Training tasks $z$ are uniform on the sphere of radius $\sqrt{d_\psi}$, or with probability $p_\text{goal}$ are $\psi$ of random dataset states (here the batch's next observations, shuffled). Targets and values use the mean of twin predictor heads. The pixel encoders, [`drq_encoder`](../../api/models.md), are DrQ-v2's convolutions with random shifts of up to 2 pixels, a linear head for $\phi$ and a two-layer MLP for $\psi$.

The [notebook](../../examples/action/tdjepa.ipynb) trains on Two-Rooms random-action pixels for 15k steps, with no rewards. It then reaches goal images in the other room 98% of the time (random actions: 3%) and solves three unseen rewards, each inferred by one linear regression, in 89% to 100% of episodes.

Choices the paper leaves open, and deviations:

- **No released code.** Network layouts follow Touati et al. (2023), which the paper cites for its architectures. Predictors embed $[x, a]$ and $[x, z]$ each with Linear, LayerNorm and Tanh, then Linear and ReLU to half width, and twin ReLU heads share these embeddings. The actor embeds $[x, z]$ and $x$ the same way. Its noise is clipped at 0.3, as in Touati et al.'s code.
- **Normalization.** The paper says encoder outputs are L2-normalized. Here they are projected to radius $\sqrt{d}$, the scale of the task sphere, as in Touati et al.'s code. Inferred tasks are projected onto the same sphere.
- **Actor loss.** The paper's pseudocode adds $+\,(T_\phi \cdot z)$ to the loss, which contradicts Algorithm 1's $-$. This implementation follows Algorithm 1. It also stops the actor loss from updating $T_\phi$, as Algorithm 1's separate updates imply.
- **Twins.** Each twin head is regressed onto the shared target and the losses are averaged over heads.
- **Goals.** The paper evaluates goal-reaching by regressing a goal reward. `goal_task` uses $\psi(g)$, or the whitened $\mathbb{E}[\psi\psi^\top]^{-1}\psi(g)$ of its Appendix D.5 when reference observations are passed. On Two-Rooms, plain $\psi(g)$ worked far better: in a tuning run at 20k steps it reached 98% of goals in the other room, against 41% for the version whitened on 1,024 dataset frames.
- **Not included:** the FlowQ-style behavior-cloning regularizer the paper adds on OGBench's narrow datasets, frame stacking (Two-Rooms images are Markov), and the TD3 fine-tuning of Section 5.4.
- **Scale.** The paper trains for 1 to 2M steps. The notebook trains for far fewer.

## Reference

Arguments and methods: [`TDJEPA`](../../api/models.md#torch_world.models.TDJEPA) in the API reference.
