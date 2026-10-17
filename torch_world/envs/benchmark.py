"""Small batched protocol for episodic benchmarks, independent of any simulator."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from torch_world.envs.rendering import resolve_quality


@dataclass
class Transition:
    """Outcome of the last action, including separate termination and time limits.

    ``success=None`` means the task has no success signal (e.g. locomotion);
    it must never be interpreted as failure. All tensors have shape ``(B,)``.
    """

    reward: Tensor
    terminated: Tensor
    truncated: Tensor
    success: Tensor | None = None
    info: Any = None

    @property
    def done(self):
        return self.terminated | self.truncated


def seed_from_generator(generator=None):
    return int(torch.randint(0, 2**31 - 1, (), generator=generator, device=getattr(generator, "device", "cpu")))


class BenchmarkEnv:
    """Implement reset/step/render, state, action_dim and transition in a subclass.

    Observations are batched tensors, either vectors or RGB pixels. Actions use
    [-1, 1]. Finished lanes are absorbing until explicit reset; no hidden autoreset.
    State is an observation snapshot, not necessarily a restorable simulator state.
    """

    def __init__(
        self, batch_size=1, size=None, device=None, max_episode_steps=200, *, quality="high", render_size=None
    ):
        self.quality = quality
        self.render_quality = resolve_quality(quality)
        size = self.render_quality.observation_size if size is None else size
        self.render_size = self.render_quality.render_size if render_size is None else render_size
        if self.render_size < 1:
            raise ValueError("render_size must be positive")
        if batch_size < 1 or size < 1 or max_episode_steps < 1:
            raise ValueError("batch_size, size and max_episode_steps must be positive")
        self.batch_size, self.size = batch_size, size
        self.device = torch.device(device or "cpu")
        self.max_episode_steps = max_episode_steps
        self._reset_episode()

    def _reset_episode(self):
        self.elapsed = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        zero = torch.zeros(self.batch_size, device=self.device)
        self.transition = Transition(zero, zero.bool(), zero.bool())

    def _action(self, action):
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action.shape != (self.batch_size, self.action_dim) or not torch.isfinite(action).all():
            raise ValueError(f"Expected finite actions of shape {(self.batch_size, self.action_dim)}")
        return action.clamp(-1, 1)

    def collect(self, num_steps, generator=None):
        """Collect T aligned transitions, without crossing reset boundaries.

        ``observations`` and ``next_observations`` are (B,T,...); ``valid`` masks
        padding after termination. ``states`` are pre-action snapshots. Unlike
        legacy pixel-world collect, every action here has a next observation.
        """
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        obs = self.reset(generator=generator)
        data = {
            k: []
            for k in (
                "observations",
                "next_observations",
                "states",
                "actions",
                "rewards",
                "terminated",
                "truncated",
                "valid",
            )
        }
        for _ in range(num_steps):
            active = ~self.transition.done
            action = (
                torch.rand(
                    self.batch_size, self.action_dim, generator=generator, device=getattr(generator, "device", "cpu")
                ).to(self.device)
                * 2
                - 1
            )
            action[~active] = 0
            data["observations"].append(obs.clone())
            data["states"].append(self.state.clone())
            obs = self.step(action)
            for key, value in (
                ("next_observations", obs),
                ("actions", action),
                ("valid", active),
                ("rewards", self.transition.reward),
                ("terminated", self.transition.terminated),
                ("truncated", self.transition.truncated),
            ):
                data[key].append(value.clone())
        out = {key: torch.stack(value, 1) for key, value in data.items()}
        if obs.ndim == 4:
            out["pixels"] = out["observations"]
        return out

    def show_env(self, *, seed=7) -> None:
        """Reset and display this world's first batch lane beside GT; leave it open."""
        from torch_world.benchmarks.visualization import show_env

        show_env(self, seed=seed)

    def show_success(self, *, policy=None, seed=7, steps=None, path=None) -> None:
        """Display a verified successful episode GIF beside GT, optionally saving to path."""
        from torch_world.benchmarks.visualization import show_success

        show_success(self, policy=policy, seed=seed, steps=steps, path=path)

    def reference_policy(self):
        """Return a fresh reference controller after reset; override for custom tasks."""
        from torch_world.envs.references import reference_policy

        return reference_policy(self)

    def render_target(self):
        """GT success-condition observation, kept separate from measured episodes."""
        return self.render(self.goal)

    def render_frame(self, state=None):
        """Presentation frame, independent of the model's observation resolution."""
        renderer = getattr(self, "render", None)
        return renderer(state) if renderer else (self.state if state is None else state).clone()

    def render_goal_frame(self):
        """GT presentation frame; never used as a surrogate model observation."""
        return self.render_target()

    @property
    def render_settings(self):
        return {"quality": self.quality, "observation_size": self.size, "render_size": self.render_size}

    def close(self):
        """Release simulator resources; subclasses may override."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
