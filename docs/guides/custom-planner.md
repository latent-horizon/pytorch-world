# Bring your own planner

A planner is any callable that turns context latents and a goal latent into an action sequence:

```python
actions = planner(model, z, goal)      # z (B, T0, ...), goal (B, ...)  ->  actions (B, horizon, action_dim)
```

It should use only `model.rollout` and `model.cost`, so it works with every `WorldModel` and with the reward objective that `run_experiment` substitutes for tasks without a goal. [`CEM`](planning.md#the-cross-entropy-method) is the built-in planner.

| argument | meaning |
|---|---|
| `model` | anything with `rollout(z, actions, **cond)` and `cost(trajectory, goal)` |
| `z` | context latents `(B, T0, ...)`, usually `model.encode(obs[:, None])` |
| `goal` | goal latents `(B, ...)` |
| returns | `(B, horizon, action_dim)`; closed-loop control executes the first action |

## An example: random shooting

Sample action sequences once, imagine them all in one batch, and keep the cheapest:

```python
import torch


class RandomShooting:
    """Sample action sequences once and keep the cheapest: CEM with a single iteration."""

    def __init__(self, horizon, action_dim, num_samples=256, bounds=(-1.0, 1.0)):
        self.horizon, self.action_dim, self.num_samples, self.bounds = horizon, action_dim, num_samples, bounds

    def __call__(self, model, z, goal, generator=None):
        B, S = z.size(0), self.num_samples
        low, high = self.bounds
        actions = torch.rand(B * S, self.horizon, self.action_dim, generator=generator, device=z.device)
        actions = low + (high - low) * actions
        traj = model.rollout(z.repeat_interleave(S, 0), actions)            # (B * S, horizon + 1, ...)
        cost = model.cost(traj, goal.repeat_interleave(S, 0)).view(B, S)
        return actions.view(B, S, self.horizon, self.action_dim)[torch.arange(B), cost.argmin(1)]
```

`rollout` takes `T0` context latents and at least `T0` actions and returns one more latent than actions. With more than one context frame, prepend the `T0 - 1` actions taken between them, as `CEM` does with `past_actions`.

## Use it

Closed loop, in any environment:

```python
from torch_world.envs import run_policy

planner = RandomShooting(horizon=8, action_dim=env.action_dim)
paths = run_policy(env, lambda obs: planner(model, model.encode(obs[:, None]), goal)[:, 0], steps=20)
```

In a benchmark, pass a factory that receives the action dimension:

```python
from torch_world.benchmarks import run_experiment

result = run_experiment("Maze-v0", planner_factory=lambda action_dim: RandomShooting(8, action_dim))
result["planner_policy"]       # "LeWM + RandomShooting"
```

The runner records `vars(planner)` with each run, so keep the settings as plain attributes.

## Conditioning

Extra model inputs, such as proprioception or embodiment ids, are keyword arguments of `rollout`. Repeat them for every sample, as `z` and `goal` are repeated, and forward them:

```python
traj = model.rollout(z.repeat_interleave(S, 0), actions, states=states.repeat_interleave(S, 0))
```
