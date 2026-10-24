import torch
from torch import Tensor

from torch_world.envs.legacy import LegacyEnvironment


class PointMass(LegacyEnvironment):
    """Batched 2D point mass in the unit square, rendered as a Gaussian blob on an ``size x size`` RGB image.

    Actions are velocities in ``[-1, 1]^2`` scaled by ``max_step``; positions are clipped to ``[low, high]^2``.
    """

    env_id = "PointMass-v0"
    action_dim = 2
    state_dim = 2
    low, high = 0.0, 1.0
    radius = 0.1

    def __init__(
        self,
        batch_size: int = 1,
        size: int = 32,
        max_step: float = 0.1,
        sigma: float = 0.06,
        device=None,
        *,
        quality="high",
        render_size=None,
        max_episode_steps=100,
        observation="rgb",
    ):
        self.batch_size, self.size, self.max_step, self.sigma = batch_size, size, max_step, sigma
        self.device = device
        self.state = torch.zeros(batch_size, 2, device=device)
        self._configure_benchmark(
            quality=quality,
            render_size=render_size,
            max_episode_steps=max_episode_steps,
            observation=observation,
            max_step=max_step,
            sigma=sigma,
        )

    def reset(self, state: Tensor | None = None, generator: torch.Generator | None = None) -> Tensor:
        if state is None:
            state = torch.rand(self.batch_size, 2, generator=generator).to(self.device)
            state = self.low + (self.high - self.low) * state
        self.state = state.clone()
        return self.render()

    def sample_goals(
        self, distance: tuple[float, float] = (0.2, 0.5), generator: torch.Generator | None = None
    ) -> Tensor:
        """Goal states at a uniform distance in ``distance`` from the current states, inside the bounds."""
        goals = self.state.clone()
        todo = torch.ones(self.batch_size, dtype=torch.bool, device=self.state.device)
        while todo.any():
            angle = torch.rand(self.batch_size, generator=generator).to(goals) * 2 * torch.pi
            radius = (torch.rand(self.batch_size, generator=generator) * (distance[1] - distance[0]) + distance[0]).to(
                goals
            )
            candidate = self.state + radius[:, None] * torch.stack([angle.cos(), angle.sin()], -1)
            inside = ((candidate >= self.low) & (candidate <= self.high)).all(-1) & todo
            goals[inside], todo = candidate[inside], todo & ~inside
        return goals

    def step(self, action: Tensor) -> Tensor:
        self.state = (self.state + self.max_step * action.clamp(-1, 1)).clamp(self.low, self.high)
        return self.render()

    def render(self, state: Tensor | None = None) -> Tensor:
        """Images ``(B, 3, size, size)`` in ``[0, 1]``: a dark blob on a light color gradient."""
        state = self.state if state is None else state
        grid = (torch.arange(self.size, device=state.device) + 0.5) / self.size
        y, x = torch.meshgrid(grid, grid, indexing="ij")
        d2 = (x - state[:, 0, None, None]).square() + (y - state[:, 1, None, None]).square()
        blob = torch.exp(-d2 / (2 * self.sigma**2))[:, None]
        background = 0.78 + 0.18 * torch.stack([x, y, 1 - x], dim=0)[None]
        ink = torch.tensor([0.10, 0.16, 0.36], device=state.device)[None, :, None, None]
        return background + blob * (ink - background)

    @staticmethod
    def position(states: Tensor) -> Tensor:
        """The planar position to reach: the point itself."""
        return states[..., :2]

    object_position = position  # what the task moves to its goal, for path-based metrics such as SPL

    def distance(self, states: Tensor, goal: Tensor) -> Tensor:
        """Distance to ``goal`` ``(B, state_dim)`` of states ``(B, ..., state_dim)``."""
        goal = goal.view(len(goal), *[1] * (states.dim() - 2), -1)
        return (self.position(states) - self.position(goal)).norm(dim=-1)

    def success(self, states: Tensor, goal: Tensor) -> Tensor:
        return self.distance(states, goal) <= self.radius

    def collect(self, num_steps: int, generator: torch.Generator | None = None, keep: float = 0.0) -> dict[str, Tensor]:
        """Random-action episodes: ``pixels`` ``(B, T, 3, size, size)``, ``actions`` ``(B, T, 2)``, ``states`` ``(B, T, 2)``.

        Each step keeps the previous action with probability ``keep``, so that episodes wander further. ``actions[:, t]``
        is applied at ``states[:, t]``.
        """
        pixels, states, actions = [self.reset(generator=generator)], [self.state], []
        for _ in range(num_steps - 1):
            action = (torch.rand(self.batch_size, 2, generator=generator) * 2 - 1).to(self.state.device)
            if keep and actions:
                kept = (torch.rand(self.batch_size, 1, generator=generator) < keep).to(self.state.device)
                action = torch.where(kept, actions[-1], action)
            actions.append(action)
            pixels.append(self.step(action))
            states.append(self.state)
        actions.append(torch.zeros_like(actions[-1]))
        return {"pixels": torch.stack(pixels, 1), "actions": torch.stack(actions, 1), "states": torch.stack(states, 1)}
