# Bring your own environment

There are two ways to add an environment:

| you want | do |
|---|---|
| a new level, layout or setting of a built-in task | [register a variant](#register-a-variant): no new code |
| new dynamics | [subclass `BenchmarkEnv`](#write-new-dynamics) |

Both end as a registered `EnvSpec`, so `make`, `list_envs` and `run_experiment` treat your task like a built-in one.

## Register a variant

`register_env` takes an `EnvSpec`: an ID, an entry point (a callable, or a lazy `"module:Class"` string), a backend and a description. `kwargs` become the constructor defaults, and `category` places the task in a [category](../environments/index.md).

```python
from torch_world.envs import EnvSpec, make, register_env, register_task

register_env(EnvSpec(
    id="MyMaze-v0",
    entry_point="torch_world.envs.puzzles:PlanningPuzzle",
    backend="torch",
    description="A custom maze with a winding corridor",
    tags=("planning", "navigation"),
    category="puzzles",
    kwargs={"task": "maze", "level": ("#######", "#A#..G#", "#.#.#.#", "#...#.#", "#######")},
))
register_task("MyMazeLong-v0", "MyMaze-v0", description="Longer planning budget", max_episode_steps=300)

env = make("MyMazeLong-v0", batch_size=4)   # constructor overrides win over the registered defaults
```

`register_task` copies an existing spec and overrides some of its `kwargs`. Use it for difficulty settings; they are not distinct physical environments. Registering an ID twice raises an error.

For a new MuJoCo scene, write a `Scene` builder (geometry, controls, initial and target coordinates, scored coordinates and tolerances) and register it with `entry_point="torch_world.envs.physics_tasks:MujocoTask"` and `kwargs={"scene": my_scene}`. The shared runner keeps timestep integration and rendering.

## Write new dynamics

Subclass `BenchmarkEnv`. The base class stores `batch_size`, `device` and `max_episode_steps`, tracks `elapsed` steps, and provides `collect`, `show_env`, `show_success` and context management.

| member | you provide |
|---|---|
| `action_dim`, `state_dim` | sizes of an action and of a state |
| `reset(state=None, *, seed=None, generator=None)` | call `self._reset_episode()`, set `self.state`, return observations `(B, ...)` |
| `step(action)` | apply actions `(B, action_dim)` in `[-1, 1]`, set `self.state` and `self.transition`, return observations |
| `render(state=None)` | observations of the current state, or of any given states |
| `state` | a snapshot `(B, state_dim)`; replace the tensor on every step, never mutate it in place |
| `transition` | a `Transition` with `reward`, `terminated`, `truncated` and `success`, each `(B,)` |

Observations are either vectors `(B, D)` or RGB images `(B, 3, H, W)` in `[0, 1]`.

Finished lanes are absorbing: after `transition.done`, a lane keeps its state and gets zero reward until the next `reset`. There is no automatic reset. Set `success=None` when the task has no success signal; it is never read as failure.

A goal-reaching task in about 50 lines:

```python
import torch

from torch_world.envs import BenchmarkEnv, EnvSpec, Transition, make, register_env


class Reach(BenchmarkEnv):
    """A point in the unit square, moved by velocity actions toward a fixed goal."""

    env_id = "Reach-v0"
    action_dim = state_dim = 2

    def __init__(self, batch_size=1, device=None, max_episode_steps=50, speed=0.05):
        super().__init__(batch_size, device=device, max_episode_steps=max_episode_steps)
        self.speed = speed
        self.goal = torch.tensor([0.8, 0.8], device=self.device).expand(batch_size, -1)
        self.state = torch.zeros(batch_size, 2, device=self.device)

    def reset(self, state=None, *, seed=None, generator=None):
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        self._reset_episode()
        if state is None:
            state = 0.4 * torch.rand(self.batch_size, 2, generator=generator)
        self.state = state.to(self.device).clone()
        return self.render()

    def step(self, action):
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device).clamp(-1, 1)
        done = self.transition.done
        state = (self.state + self.speed * action).clamp(0, 1)
        self.state = torch.where(done[:, None], self.state, state)   # finished lanes stay put
        self.elapsed += ~done
        success = self.success(self.state, self.goal)
        self.transition = Transition(
            reward=-self.distance(self.state, self.goal) * ~done,
            terminated=done | success,
            truncated=~success & (self.elapsed >= self.max_episode_steps),
            success=success,
        )
        return self.render()

    def render(self, state=None):
        return (self.state if state is None else state).clone()      # state-vector observations

    def distance(self, states, goal):
        return (states - goal).norm(dim=-1)

    def success(self, states, goal):
        return self.distance(states, goal) < 0.05


register_env(EnvSpec("Reach-v0", Reach, "torch", "Move a point to a fixed goal", category="navigation"))
```

## Run it

```python
from torch_world.benchmarks import run_experiment

with make("Reach-v0", batch_size=4) as env:
    data = env.collect(8)        # observations, next_observations, actions, rewards, valid, ... each (4, 8, ...)
    result = run_experiment(env)
```

`run_experiment` trains the default LeWM on random-action data and compares LeWM + CEM with random actions. It decides how to plan from what the environment offers:

| the environment has | planning objective |
|---|---|
| `goal`, `distance(states, goal)`, `success(states, goal)` and `render(state)` | reach the encoded goal, scored by `success` at the end of the episode |
| none of these | a reward head on frozen latents, scored by the summed reward |

Only expose the goal members when states can really be scored and rendered.

## Optional members

| member | enables |
|---|---|
| `render_frame()`, `render_goal_frame()` | RGB presentation frames `(B, 3, H, W)` for previews and episode GIFs, even when observations are vectors |
| `render_target()` | the GT observation at model resolution; the default renders `self.goal` |
| `reference_policy()` | a fresh controller after reset, so `show_success()` can replay a verified episode |
| `close()` | release simulator resources; called on exit from `with` |

## Contribute it upstream

Built-in environments live in `torch_world/envs/` and are registered in `catalog.py`. Include tests for deterministic resets, known dynamics or solutions, action bounds, finite observations, terminal padding and cleanup, plus a reference controller and a notebook in `examples/envs/<category>/`. `puzzles.py`, `physics_scenes.py` with `physics_tasks.py`, and `newton.py` are complete examples. See [Contributing](../contributing.md).
