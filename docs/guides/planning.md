# Planning

A trained `WorldModel` can imagine the outcome of any action sequence. Planning searches for the sequence whose imagined final latent is closest to the latent of a goal.

## Rollout and cost

Both come with `WorldModel`, built on your `predict`:

```python
z = model.encode(pixels[:, :1])        # (B, 1, D) context latents
traj = model.rollout(z, actions)       # actions (B, T, A)  ->  latents (B, T + 1, D)
cost = model.cost(traj, goal)          # goal (B, D)        ->  (B,)
```

`rollout` predicts one step at a time and feeds each prediction back in, keeping the last `model.history` latents. `cost` is the mean squared distance between the final latent and the goal; override it for another metric (RoboJEPA uses L1).

## The cross-entropy method

[`CEM`](../api/planning.md) keeps a diagonal Gaussian over action sequences. Each iteration it samples `num_samples` sequences, rolls them all out in one batch, and refits the Gaussian to the `num_elites` cheapest.

```python
from torch_world.planning import CEM

planner = CEM(horizon=5, action_dim=2, num_samples=512, num_elites=48, iterations=10, bounds=(-1, 1))
actions = planner(model, z, goal)      # (B, horizon, action_dim), the final mean
```

| argument | meaning |
|---|---|
| `horizon` | length of the planned action sequence |
| `num_samples`, `num_elites` | population size and number kept per iteration |
| `iterations` | refits of the Gaussian |
| `std` | initial standard deviation (the mean starts at zero) |
| `bounds` | clamp samples to the action space |

The batch dimension is preserved: `B` problems are planned in parallel, `B × num_samples` rollouts per iteration.

### Context and conditioning

With more than one context frame, pass the actions taken between them as `past_actions` `(B, T0 - 1, A)`. Extra keyword arguments are forwarded to `model.rollout`, repeated for every sample. That is how DINO-WM gets proprioception and RoboJEPA gets states and embodiment ids:

```python
planner(model, z, goal, states=env.state[:, None], embodiment=ids)
```

Pass a `torch.Generator` as `generator` for reproducible plans.

## MPPI: planning for reward

[TD-MPC2](../models/tdmpc/tdmpc2.md) plans for the highest return instead of toward a goal. [`MPPI`](../api/planning.md) scores action sequences with `model.estimate_value`: predicted rewards over the horizon plus a learned value at its end. Each iteration it adds `num_pi_trajs` sequences from the model's policy prior to Gaussian samples, and refits the Gaussian to the `num_elites` best, weighted by `softmax(temperature * value)`. The plan it returns is one elite drawn with those weights.

```python
from torch_world.planning import MPPI

planner = MPPI(horizon=3, action_dim=2)                          # the paper's settings
plan = planner(model, model.encode(obs[:, None]), explore=True)  # (B, horizon, action_dim), in [-1, 1]
```

`MPPI` remembers its last mean and starts the next call from it, shifted by one step: call `planner.reset()` between episodes. `explore=True` adds noise with the final std, as when collecting data.

## Closed loop

Model-predictive control: plan, execute the first action, observe, plan again. [`run_policy`](../api/envs.md) runs any `policy(observation) -> action` in an environment and returns the visited states.

```python
from torch_world.envs import run_policy

with torch.no_grad():
    goal = model.encode(env.render(goal_states)[:, None])[:, 0]

paths = run_policy(env, lambda obs: planner(model, model.encode(obs[:, None]), goal)[:, 0], steps=20)
distance = (paths - goal_states[:, None]).norm(dim=-1)   # (B, steps + 1)
```

## Evaluating goal reaching

[`goal_reaching`](../api/evaluation.md) runs several policies from the same starts toward the same goals and scores them with the environment's own `distance` and `success`:

```python
from torch_world.evaluation import goal_reaching, random_policy

env.reset()
start, goals = env.state.clone(), env.sample_goals((0.2, 0.5))
with torch.no_grad():
    goal = model.encode(env.render(goals)[:, None])[:, 0]

result = goal_reaching(
    env,
    {"LeWM + CEM": lambda obs: planner(model, model.encode(obs[:, None]), goal)[:, 0], "random": random_policy(env)},
    start, goals, steps=20,
)
result.rates()        # success rate per policy, e.g. {"LeWM + CEM": ..., "random": ...}
result.distances      # per policy, (B, steps + 1) distance to the goal at every step
```

`plots.distances(result.distances, radius=env.radius)` and `plots.trajectories` draw the result; see [Figures](plots.md).
