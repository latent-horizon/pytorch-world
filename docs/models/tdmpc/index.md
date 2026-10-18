# TD-MPC

**Temporal-difference model predictive control** learns a world model for control alone. There is no decoder and no reconstruction: an encoder maps observations to latents, and latent dynamics are trained to predict the encoding of the next observation, the reward and, through Q-functions learned by temporal-difference learning, the value of what comes after. The latent only has to carry what matters for return. A planner searches action sequences inside the model, scoring each by imagined rewards over a short horizon plus a learned value at its end, and a policy prior proposes candidate sequences and bootstraps those values.

All models in this family live in `torch_world.models.tdmpc` and are exported from `torch_world.models`. They subclass `WorldModel`, so `encode`, `predict` and `rollout` work as for every world model; they plan for reward with [`MPPI`](../../guides/planning.md#mppi-planning-for-reward), not toward a goal latent.

<div class="grid cards" markdown>

-   **[TD-MPC2](tdmpc2.md)**

    ---

    SimNorm latents, an ensemble of Q-functions with discrete regression, and a tanh-Gaussian policy prior; plans with MPPI.

    reward-driven · TD learning · MPPI

</div>

## How it differs from JEPA world models

| | JEPA world models | TD-MPC |
|---|---|---|
| learns from | offline trajectories (frames + actions) | trajectories with rewards, usually collected online |
| latent trained by | next-embedding prediction + an anti-collapse term | next-latent consistency + reward and value prediction |
| plans toward | a goal latent (`cost`) | the highest return (`estimate_value`) |
| planner | `CEM` | `MPPI`, with policy-prior proposals |

## Papers and code

| model | paper | reference code |
|---|---|---|
| `TDMPC2` | [Hansen et al., 2024](https://arxiv.org/abs/2310.16828) | [nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2) |
