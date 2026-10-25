# Dreamer

**Dreamer** agents learn a world model by reconstructing what they see, and learn to act inside it. A recurrent state-space model (RSSM) compresses each frame, given everything before it, into a model state: a deterministic recurrent state plus a sampled stochastic state. A decoder rebuilds the frame from that state, heads predict the reward and whether the episode continues, and a prior learns to predict the next stochastic state without seeing the frame. An actor and a critic then train on trajectories the model imagines from replayed states, never touching the environment while they learn.

All models in this family live in `torch_world.models.dreamer` and are exported from `torch_world.models`. They subclass `WorldModel`, so `encode`, `predict` and `rollout` work as for every world model, and [`CEM`](../../guides/planning.md) can plan on the model state toward a goal image; the actor is a second way to act, with no search.

<div class="grid cards tw-models">

-   **[DreamerV3](dreamerv3.md)**

    RSSM with categorical latents and a pixel decoder, KL balancing with free bits, symlog two-hot rewards and values, and an actor-critic learned in imagination.

    `reconstruction` `RSSM` `actor-critic`
    { .tw-chips }

</div>

## How it differs from JEPA and TD-MPC world models

| | JEPA world models | TD-MPC | Dreamer |
|---|---|---|---|
| latent trained by | next-embedding prediction + an anti-collapse term | next-latent consistency + reward and value prediction | pixel reconstruction + KL between prior and posterior |
| decoder | none | none | yes: imagined states can be looked at |
| state | one embedding per frame | one latent per frame | recurrent state + stochastic categoricals, filtered over time |
| acts with | a planner toward a goal latent | MPPI for return | an actor learned in imagination (or a planner on the state) |

## Papers and code

| model | paper | reference code |
|---|---|---|
| `DreamerV3` | [Hafner et al., 2023](https://arxiv.org/abs/2301.04104) | [danijar/dreamerv3](https://github.com/danijar/dreamerv3) |
