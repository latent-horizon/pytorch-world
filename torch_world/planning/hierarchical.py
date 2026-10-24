"""Top-down planning with two-level world models such as HWM and H-JEPA."""

import copy

import torch
from torch import Tensor

from torch_world.planning.cem import _repeat


class GradientPlanner:
    """Gradient descent on action sequences through a differentiable world-model rollout, as H-JEPA plans.

    ``num_samples`` candidates per batch element start from ``N(0, std^2)``, one of them at the mean, and are optimized
    with Adam on ``model.cost`` for ``iterations`` steps, clipped to ``bounds`` after each step. The cheapest candidate
    seen is returned. Only the actions receive gradients; put the model in eval mode first.
    """

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        num_samples: int = 32,
        iterations: int = 30,
        lr: float = 0.1,
        std: float = 1.0,
        bounds: tuple[float, float] | None = None,
    ):
        self.horizon, self.action_dim = horizon, action_dim
        self.num_samples, self.iterations, self.lr, self.std, self.bounds = num_samples, iterations, lr, std, bounds

    def __call__(self, model, z: Tensor, goal: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Plan ``(B, horizon, action_dim)`` from context latents ``z`` ``(B, T0, ...)`` toward ``goal`` ``(B, ...)``."""
        B, S = z.size(0), self.num_samples
        shape = (B, S, self.horizon, self.action_dim)
        init = self.std * torch.randn(shape, device=z.device, dtype=z.dtype, generator=generator)
        init[:, 0] = 0
        if self.bounds is not None:
            init = init.clamp(*self.bounds)
        actions = init.flatten(0, 1).requires_grad_(True)
        z, goal = _repeat(z.detach(), S), _repeat(goal.detach(), S)
        optimizer = torch.optim.Adam([actions], lr=self.lr)
        best, best_cost = actions.detach().clone(), torch.full((B * S,), float("inf"), device=z.device)
        with torch.enable_grad():
            for step in range(self.iterations + 1):
                cost = model.cost(model.rollout(z, actions), goal)
                with torch.no_grad():
                    better = cost < best_cost
                    best[better], best_cost[better] = actions[better], cost[better]
                if step == self.iterations:
                    break
                optimizer.zero_grad()
                actions.grad = torch.autograd.grad(cost.sum(), actions)[0]
                optimizer.step()
                if self.bounds is not None:
                    with torch.no_grad():
                        actions.clamp_(*self.bounds)
        index = best_cost.view(B, S).argmin(1)
        return best.view(shape)[torch.arange(B, device=z.device), index]


class HierarchicalPlanner:
    """Two-level top-down planning: the high level picks subgoals, the low level plans actions to reach the next one.

    ``model`` is a two-level world model (``HWM``, ``HJEPA``): itself the low-level ``WorldModel``, ``model.high`` the
    high-level one over latent macro-actions, ``model.lift`` maps low-level latents to high-level latents, and
    ``model.stride`` counts low-level steps per high-level step. Each call:

    1. lifts the current latent and the goal latent to the high level, and ``high(model.high, ...)`` searches
       macro-actions whose rollout ends at the goal;
    2. the first predicted high-level latent becomes the subgoal; when the high horizon is 1 the goal itself is,
       as H-JEPA anchors the last subgoal to the goal;
    3. ``low(...)`` searches primitive actions, usually ``model.stride`` of them, whose last predicted latent, lifted,
       matches the subgoal under ``model.high.cost``.

    ``high`` and ``low`` are planners with the ``CEM`` signature (``CEM``, ``GradientPlanner``); ``high`` plans
    standardized macro-actions, so ``bounds=(-2, 2)`` matches the papers' clipping to training statistics. With
    ``shrink_every``, the high horizon shrinks by one every ``shrink_every`` calls down to ``min_horizon``, as the
    papers shorten it while the episode's budget runs out; call ``reset`` between episodes. ``subgoals`` keeps the
    last high-level rollout ``(B, H, D)``.
    """

    def __init__(self, high, low, shrink_every: int | None = None, min_horizon: int = 1):
        self.high, self.low, self.shrink_every, self.min_horizon = high, low, shrink_every, min_horizon
        self.calls, self.subgoals = 0, None

    def reset(self):
        self.calls = 0

    def horizon(self) -> int:
        if self.shrink_every is None:
            return self.high.horizon
        return max(self.min_horizon, self.high.horizon - self.calls // self.shrink_every)

    @torch.no_grad()
    def __call__(self, model, z: Tensor, goal: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Plan ``(B, low.horizon, A)`` from low-level latents ``z`` ``(B, T0, D)`` toward ``goal`` ``(B, D)``."""
        H = self.horizon()
        high = self.high if H == self.high.horizon else copy.copy(self.high)
        high.horizon = H
        zh, gh = model.lift(z[:, -1:]), model.lift(goal[:, None])[:, 0]
        macro = high(model.high, zh, gh, generator=generator)
        self.subgoals = model.high.rollout(zh, macro)[:, 1:]
        subgoal = gh if H == 1 else self.subgoals[:, 0]
        self.calls += 1
        return self.low(_TowardSubgoal(model), z, subgoal, generator=generator)


class _TowardSubgoal:
    """The low level of ``model``, scored by the distance of its last latent, lifted, to a high-level subgoal."""

    def __init__(self, model):
        self.model = model

    def rollout(self, z: Tensor, actions: Tensor) -> Tensor:
        return self.model.rollout(z, actions)

    def cost(self, z: Tensor, subgoal: Tensor) -> Tensor:
        return self.model.high.cost(self.model.lift(z[:, -1:]), subgoal)
