"""Benchmark protocol for the library's original pixel and MuJoCo environments.

The original simulation and data-collection APIs remain unchanged. These
recipes add fixed goals, time limits, aligned transitions and high-resolution
presentation frames to the same underlying implementations.
"""

from copy import deepcopy
from functools import cached_property
from importlib import import_module

import torch

from torch_world.envs.benchmark import BenchmarkEnv, Transition
from torch_world.envs.rendering import resolve_quality

_MODULES = {"PointMass": "point_mass", "Macaw": "macaw", "PushT": "push_t", "RobotCube": "robot_cube"}


class LegacyEnvironment:
    """Connect original worlds to experiments without changing their ``collect`` API.

    ``show_env()``, ``show_success()`` and ``run_experiment(world, ...)`` share
    one cached benchmark view of this simulator. Use ``as_benchmark()`` when
    aligned transitions, seeded resets or explicit episode limits are needed.
    """

    def _configure_benchmark(
        self, *, quality, render_size, max_episode_steps, observation, presentation_camera="overview", **options
    ):
        preset = resolve_quality(quality)
        if render_size is not None and render_size < 1:
            raise ValueError("render_size must be positive")
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        if observation not in {"rgb", "state"}:
            raise ValueError("observation must be rgb or state")
        if presentation_camera not in {"overview", "task"}:
            raise ValueError("presentation_camera must be overview or task")
        self._benchmark_settings = {
            "quality": quality,
            "render_size": preset.render_size if render_size is None else render_size,
            "max_episode_steps": max_episode_steps,
            "observation": observation,
            "presentation_camera": presentation_camera,
            "options": options,
        }

    @cached_property
    def _benchmark_env(self):
        from torch_world.envs.registry import get_spec

        spec = get_spec(self)
        env = LegacyTask(
            spec.kwargs["environment"],
            batch_size=self.batch_size,
            size=self.size,
            device=self.device,
            **self._benchmark_settings,
            _world=self,
        )
        env.spec = spec
        return env

    def as_benchmark(self) -> BenchmarkEnv:
        """Return the shared episode view; it operates on this world's actual state."""
        return self._benchmark_env

    def show_env(self, *, seed=7) -> None:
        """Reset and display the environment beside GT at presentation resolution."""
        self.as_benchmark().show_env(seed=seed)

    def show_success(self, *, policy=None, seed=7, steps=None, path=None) -> None:
        """Display a verified successful episode GIF beside GT, optionally saving it."""
        self.as_benchmark().show_success(policy=policy, seed=seed, steps=steps, path=path)

    def close(self):
        """Release renderers without constructing a benchmark view."""
        for renderer in getattr(self, "renderers", {}).values():
            renderer.close()
        if hasattr(self, "renderers"):
            self.renderers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class LegacyTask(BenchmarkEnv):
    """Episode protocol shared by named original worlds and registered recipes.

    ``options`` forwards original constructor settings, e.g. RobotCube's robot,
    camera or asset-cache root. Observation and presentation sizes are separate.
    State vectors retain the original meaning; robot states are not full physics
    checkpoints. PushT uses a fixed, symmetric translation task; other recipes
    retain seeded random starts and use a fixed goal throughout an experiment.
    """

    def __init__(
        self,
        environment,
        *,
        batch_size=1,
        size=None,
        quality="high",
        render_size=None,
        device=None,
        max_episode_steps=100,
        observation="rgb",
        options=None,
        presentation_camera="overview",
        _world=None,
    ):
        super().__init__(batch_size, size, device, max_episode_steps, quality=quality, render_size=render_size)
        self.settings = deepcopy(
            {
                "environment": environment,
                "batch_size": batch_size,
                "size": self.size,
                "quality": quality,
                "render_size": self.render_size,
                "device": self.device,
                "max_episode_steps": max_episode_steps,
                "observation": observation,
                "options": dict(options or {}),
                "presentation_camera": presentation_camera,
            }
        )
        if observation not in {"rgb", "state"}:
            raise ValueError("observation must be rgb or state")
        self.environment, self.observation = environment, observation
        if presentation_camera not in {"overview", "task"}:
            raise ValueError("presentation_camera must be overview or task")
        self.presentation_camera = presentation_camera
        self._camera = None
        cls = getattr(import_module(f"torch_world.envs.{_MODULES[environment]}"), environment)
        options = dict(options or {})
        self.world = (
            _world
            if _world is not None
            else cls(
                batch_size=batch_size, size=self.size if observation == "rgb" else 1, device=self.device, **options
            )
        )
        self.action_dim, self.state_dim = self.world.action_dim, self.world.state_dim
        self._physics = hasattr(self.world, "model")
        self._presentation = None
        self.initial = None
        if self._physics:
            model = self.world.model
            self.world.render_shadows = self.render_quality.shadows
            model.vis.global_.offwidth = max(self.size, self.render_size, 512)
            model.vis.global_.offheight = max(self.size, self.render_size, 512)
            model.vis.quality.offsamples = self.render_quality.samples
        else:
            self._presentation = cls(batch_size=batch_size, size=self.render_size, device=self.device, **options)
        if environment == "RobotCube":
            if presentation_camera == "overview":
                self._camera = self.world.mujoco.MjvCamera()
                self._camera.lookat[:] = (0.4, 0, 0.3) if self.world.robot == "ur10e" else (0.18, 0, 0.13)
                self._camera.distance = 1.8 if self.world.robot == "ur10e" else 0.65
                self._camera.azimuth, self._camera.elevation = 135, -25
            (x0, x1), (y0, y1) = self.world.config.workspace
            x, y = x0 + 0.35 * (x1 - x0), y0 + 0.3 * (y1 - y0)
            target = [x, y, self.world.hover, 1, x, y, self.world.half]
        elif environment == "PushT":
            self.initial = torch.tensor([0, 0.12, 0, 0.06, 0], device=self.device).expand(batch_size, -1)
            target = [0, -0.085, 0, -0.10, 0]  # the pusher rests on the bar it pushed
        else:
            target = [0.75, 0.25]
        self.goal = torch.tensor(target, device=self.device).expand(batch_size, -1).clone()

    @property
    def state(self):
        return self.world.state

    def reset(self, state=None, *, seed=None, generator=None):
        if seed is not None and generator is not None:
            raise ValueError("Pass seed or generator, not both")
        generator = torch.Generator().manual_seed(seed) if seed is not None else generator
        self._reset_episode()
        pixels = self.world.reset(self.initial if state is None else state, generator=generator)
        return self.state.clone() if self.observation == "state" else pixels

    def step(self, action):
        action = self._action(action)
        active = ~self.transition.done
        pixels = None
        if active.any():
            # The adapter has only a shared time limit, so all lanes finish together.
            pixels = self.world.step(action)
        self.elapsed += active
        self.transition = Transition(
            (-self.distance(self.state, self.goal) - 0.001 * action.square().sum(-1)) * active,
            torch.zeros_like(active),
            self.elapsed >= self.max_episode_steps,
            self.success(self.state, self.goal),
        )
        return self.render() if self.observation == "state" or pixels is None else pixels

    def distance(self, states, goal):
        return self.world.distance(states, goal)

    def success(self, states, goal):
        reached = self.world.success(states, goal)
        if self.environment == "RobotCube" and self.world.task == "pick":
            reached &= (states[..., 3] > 0.9) & (states[..., 2] > self.world.hover * 0.8)
        return reached

    def render(self, state=None):
        if self.observation == "state":
            return (self.state if state is None else state).clone()
        return self.world.render(state)

    def render_frame(self, state=None):
        if self._physics:
            return self.world.render(state, size=self.render_size, camera=self._camera)
        if self.environment == "Macaw":
            phase = self.world.phase if state is None else None
            return self._presentation.render(self.state if state is None else state, phase=phase)
        return self._presentation.render(self.state if state is None else state)

    def render_goal_frame(self):
        return self.render_frame(self.goal)

    @property
    def render_settings(self):
        return {**super().render_settings, "presentation_camera": self.presentation_camera}

    def reference_policy(self):
        if self.environment == "RobotCube":
            return lambda obs: self.world.scripted(self.goal)
        if self.environment == "PushT":
            return lambda obs: torch.stack(
                (
                    torch.zeros(self.batch_size, device=self.device),
                    ((self.goal[:, 3] - self.state[:, 3]) / self.world.max_step).clamp(-0.5, 0.5),
                ),
                -1,
            )
        return lambda obs: ((self.goal - self.state) / self.world.max_step).clamp(-0.35, 0.35)

    def close(self):
        for renderer in getattr(self.world, "renderers", {}).values():
            renderer.close()
        if self._physics:
            self.world.renderers.clear()
