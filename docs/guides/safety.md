# Safety filters

> Nakamura, Peters and Bajcsy, *Generalizing Safety Beyond Collision-Avoidance via Latent-Space Reachability Analysis*, 2025. [Paper](https://arxiv.org/abs/2502.00935) · [Code](https://github.com/CMU-IntentLab/latent-safety) · [Notebook](../examples/safety/latent_safety_filter.ipynb)

`torch_world.safety` puts a safety layer on top of any [`WorldModel`](../api/models.md): it does not change the model and does not care how it was trained. Safety is defined by examples in the model's latent space, made predictive by Hamilton-Jacobi reachability in imagination, and enforced by a filter in front of any policy: a planner, a learned policy or a person at a teleoperation console.

Three pieces, each usable alone:

| piece | what it is | trained on |
|---|---|---|
| [`FailureMargin`](../api/safety.md#torch_world.safety.FailureMargin) | a classifier read as a signed margin $\ell(z)$, negative on failures | encoded frames labeled safe or failure |
| [`SafetyValue`](../api/safety.md#torch_world.safety.SafetyValue) | the safety value $V(z)$ and the safety policy $\pi(z)$ | imagined transitions of the frozen world model |
| [`SafetyFilter`](../api/safety.md#torch_world.safety.SafetyFilter) | the least-restrictive filter in front of a task policy | nothing: it combines the two above |

## Failure margin

Failure is whatever the labels say: a collision, a spill, a block knocked over, a robot in a forbidden zone. An MLP on the frozen latents scores each frame, and $\ell(z) = \tanh(s(z))$ is its margin, so the failure set is $\{z : \ell(z) < 0\}$. Two hinges push safe latents above $\delta$ and failures below $-\delta$, each averaged over its own class, so that rare failures count as much as the safe majority:

$$
\mathcal{L}(\mu) = \frac{1}{N_\text{safe}} \sum_{z \in \mathcal{D}_\text{safe}} \operatorname{ReLU}\big(\delta - s_\mu(z)\big) + \frac{1}{N_\text{fail}} \sum_{z \in \mathcal{D}_\text{fail}} \operatorname{ReLU}\big(\delta + s_\mu(z)\big)
$$

```python
from torch.utils.data import DataLoader, TensorDataset

from torch_world.safety import FailureMargin
from torch_world.utils import fit

with torch.no_grad():
    latents = model.encode(pixels)                    # (N, T, D) from the frozen world model
margin = FailureMargin(D, hidden=(256, 256))           # delta = 0.75, the reference code's value
loader = DataLoader(TensorDataset(latents.flatten(0, 1), failure.flatten()), batch_size=512, shuffle=True)
fit(margin, loader, epochs=5, lr=1e-3, inputs=lambda b: b)   # {"loss", "safe_loss", "failure_loss", "accuracy"}
l = margin.margin(z)                                  # (B,), negative = failure
```

Latents of any shape work: pass `latent_shape=(196, 384)` for patch tokens, and they are flattened.

## Safety value by reachability in imagination

$\ell$ only says whether a latent *is* a failure. Whether failure *can still be avoided* is the safety value: the lowest margin the system will ever reach when it acts as safely as it can. It satisfies the discounted avoid Bellman equation of HJ reachability, here on latents $\hat z' = f(z, a)$ imagined by the world model:

$$
V(z) = (1 - \gamma)\,\ell(z) + \gamma \min\Big\{ \ell(z),\ \max_{a} V\big(f(z, a)\big) \Big\}
$$

$\gamma < 1$ makes the backup a contraction; as $\gamma \to 1$, $V$ approaches the undiscounted value $\min_t \ell(z_t)$ along the best trajectory. With continuous actions it is learned with DDPG as in the paper: a critic $Q(z, a)$ regresses the backup with target networks $Q', \pi'$ (Polyak rate $\tau$), and an actor $\pi(z)$ maximizes it. $\pi$ is the safety policy and $V(z) = Q(z, \pi(z))$.

```python
from torch_world.safety import SafetyValue, fit_safety_value

value = SafetyValue(D, action_dim=2, hidden=(512, 512, 512, 512), gamma=0.9999)
log = fit_safety_value(value, model, margin, starts=latents.flatten(0, 1), iterations=60_000, warmup=5_000)
value.value(z)      # (B,) safety value
value.action(z)     # (B, 2) safety policy
```

`fit_safety_value` freezes the world model and the margin, and runs `envs` imagined episodes in parallel. Each starts from a latent of `starts` (encoded dataset frames) and runs `horizon` steps of the model (the paper's 25, to limit drift) before it restarts. Transitions go to a replay buffer; one update follows every `envs` transitions, as in the reference code (8 per update). Exploration and warm-up follow the reference code too: the actor's action with Gaussian noise, replaced by a random action whenever the critic deems that one safe, and `warmup` iterations of random actions with $\gamma = 0$, which fit $Q$ to $\ell$ before the backup starts. The world model steps one latent at a time through [`imagine_step`](../api/safety.md#torch_world.safety.imagine_step) (`model.rollout` from a single context latent).

## The filter

The least-restrictive filter lets the task policy act as long as the latent it leads to keeps a safety value above a threshold $\epsilon$, and hands control to $\pi$ otherwise:

$$
a_t = \begin{cases} a^\text{task}_t & \text{if } V\big(f(z_t, a^\text{task}_t)\big) > \epsilon \\ \pi(z_t) & \text{otherwise} \end{cases}
$$

```python
from torch_world import evaluation
from torch_world.safety import SafetyFilter

safety = SafetyFilter(model, value, threshold=0.4)    # the paper's 0.4 in simulation, 0.3 on hardware
a, overridden = safety(z, task_actions)               # (B, A), (B,)

policy = safety.wrap(evaluation.goal_policy(model, planner, goal_images))   # observation -> action
run = evaluation.goal_reaching(env, {"CEM + filter": policy}, start, goal, steps=60)
policy.overrides                                      # per step, which episodes the safety policy took over
```

$\epsilon > 0$ leaves room for errors of the learned value and of the predicted latent. `rule="q"` checks $Q(z_t, a^\text{task}_t) > \epsilon$ instead, which skips the world-model call; the critic was trained on the same imagined transitions, so the two agree when it fits.

## In the notebook

[safety/latent_safety_filter](../examples/safety/latent_safety_filter.ipynb) draws two red pools on PLDM's Two-Rooms floor, trains PLDM on random-action episodes, the margin on frames labeled by whether the point is inside a pool, and the value function in PLDM's imagination. A goal-reaching CEM planner that knows nothing of the pools then runs with and without the filter.

The run uses 100 tasks, 60 steps each, with success counted when the goal is reached at any step. Rates carry 95% Wilson intervals. "Failure" means the point entered a pool at any step.

| policy | goal reached | failure | safe success |
|---|---|---|---|
| CEM | 94% [88, 97] | 81% [72, 87] | 17% [11, 26] |
| CEM + filter (ε = 0.4) | 53% [43, 62] | 78% [69, 85] | 17% [11, 26] |
| random | 3% [1, 8] | 78% [69, 85] | 0% [0, 4] |
| random + filter | 1% [0, 5] | 52% [42, 62] | 0% [0, 4] |

The margin and the value function come out right. The margin catches 100% of held-out failures with 1% false alarms. The unsafe set of `V` matches the pools, which is the exact answer for a point that can stop at once. The safety policy never steps into a pool. The filter, though, does not keep CEM out of the pools, and it halves goal reaching. Thresholds from 0 to 0.6 and the `Q` rule all leave failures at 72–80%.

The notebook traces the cause to the world model. One step from the edge of a pool, `V` of the encoded real next frame flags 100% of the moves that enter the pool, but `V` of PLDM's *predicted* next latent flags only 65%. A planner that keeps pushing toward the door eventually finds one of the moves the prediction gets wrong. The filter can be no better than the one-step model it consults, and here that model was trained on random-action data alone.

## Differences from the paper

- **World models.** The paper uses a Dreamer RSSM in simulation and DINO-WM on hardware. Here any `WorldModel` works; the notebook uses PLDM. The value function sees one latent at a time and imagines with `model.rollout` from that latent alone, so for models with a history of several frames (DINO-WM, LeWM) the imagined step uses a one-frame context. The RSSM's recurrent state carries the history that this does not.
- **When the margin is trained.** In simulation the paper trains the classifier jointly with the world model; on hardware, after it, on frozen latents. Here it is always the latter.
- **Margin details.** The paper does not report $\delta$; the default is the reference code's 0.75. The reference computes the backup with the margin of the next latent as the per-step reward; this follows the paper's equation, with $\ell(z)$ of the current latent.
- **Value function.** DDPG only (the paper's continuous-action setting); the DDQN variant for discrete actions is not included, and neither is the SQRL baseline. The paper's DDPG uses four hidden layers of 512 (the default here); the reference code's configuration uses three of 128. The world model is treated as deterministic: for stochastic models, the expectation over $\hat z'$ is estimated with one sample per transition.
- **Filter.** The paper's rule. `rule="q"` is an addition.
