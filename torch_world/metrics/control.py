import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Rate:
    """A success rate over ``n`` episodes with its Wilson score interval ``[low, high]``."""

    value: float
    low: float
    high: float
    n: int

    def __float__(self) -> float:
        return self.value

    def __str__(self) -> str:
        return f"{self.value:.0%} [{self.low:.0%}, {self.high:.0%}]"


def success_rate(success: Tensor, confidence: float = 0.95) -> Rate:
    """Fraction of successful episodes ``(N,)`` with its Wilson score interval, which stays inside ``[0, 1]`` and is
    reliable at the few dozen episodes robot evaluations afford (16 episodes at 50% give ``[28%, 72%]``).

    Wilson, "Probable Inference, the Law of Succession, and Statistical Inference", JASA 1927.
    """
    n, k = success.numel(), success.float().sum().item()
    p = k / n
    z = math.sqrt(2) * torch.erfinv(torch.tensor(confidence, dtype=torch.float64)).item()
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z / (1 + z**2 / n) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return Rate(p, max(0.0, center - half), min(1.0, center + half), n)


def first_success(success: Tensor) -> Tensor:
    """Index of the first successful step of each episode ``(N, T)``, or ``-1`` for episodes that never succeed."""
    hit = success.bool()
    first = hit.int().argmax(1)
    return torch.where(hit.any(1), first, torch.full_like(first, -1))


def progress(distances: Tensor) -> Tensor:
    """Fraction of the initial distance to the goal closed by the end of each episode ``(N, T)``: 1 at the goal,
    0 when no closer than at the start, negative when farther, and NaN for episodes that start at the goal."""
    start = distances[:, 0]
    return (1 - distances[:, -1] / start).where(start > 0, torch.nan)


def path_length(positions: Tensor) -> Tensor:
    """Length of each path of positions ``(N, T, D)``."""
    return (positions[:, 1:] - positions[:, :-1]).norm(dim=-1).sum(1)


def spl(success: Tensor, positions: Tensor, goal: Tensor) -> float:
    """Success weighted by path length: the mean of ``success * shortest / max(path, shortest)`` over episodes, with
    the straight line from the start position to the ``goal`` position ``(N, D)`` as the shortest path. A success by
    a detour twice too long counts one half.

    Anderson et al., "On Evaluation of Embodied Navigation Agents", 2018.
    Paper: https://arxiv.org/abs/1807.06757
    """
    shortest = (goal - positions[:, 0]).norm(dim=-1)
    taken = path_length(positions)
    return (success.float() * shortest / torch.maximum(taken, shortest).clamp_min(1e-9)).mean().item()


def smoothness(actions: Tensor) -> Tensor:
    """Mean size of the change between consecutive actions ``(N, T, A)`` per episode; lower is smoother. Jerky
    commands that flip direction every step, typical of sampling-based planners, score high."""
    return (actions[:, 1:] - actions[:, :-1]).norm(dim=-1).mean(1)
