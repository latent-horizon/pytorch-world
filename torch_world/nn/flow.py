"""Rectified-flow helpers: time ``0`` is Gaussian noise and time ``1`` is data.

Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023. Paper: https://arxiv.org/abs/2210.02747
"""

import math
from collections.abc import Callable

import torch
from torch import Tensor


def time_embedding(time: Tensor, dim: int) -> Tensor:
    """Sinusoidal embedding ``(B, dim)`` of flow times ``(B,)``."""
    frequency = torch.exp(-math.log(10000) * torch.arange(dim // 2, device=time.device) / (dim // 2))
    phase = time[..., None] * frequency
    return torch.cat((phase.sin(), phase.cos()), dim=-1)


def interpolate(noise: Tensor, target: Tensor, time: Tensor) -> Tensor:
    """Point at ``time`` on the straight path from ``noise`` to ``target``; ``time`` broadcasts from the left."""
    return torch.lerp(noise, target, time.reshape(time.shape + (1,) * (target.dim() - time.dim())))


def velocity_target(noise: Tensor, target: Tensor) -> Tensor:
    """Constant velocity of the straight path, the regression target of flow matching."""
    return target - noise


def sample_time(
    shape: tuple[int, ...] | int,
    *,
    device=None,
    dtype=None,
    generator: torch.Generator | None = None,
    alpha: float | None = None,
    logit_normal: tuple[float, float] | None = None,
) -> Tensor:
    """Training flow times in ``[0, 1]``: ``U(0, 1)`` by default.

    ``alpha`` draws ``1 - Beta(alpha, 1)``; ``alpha > 1`` concentrates times near noise, as DreamZero-Flash does for
    video (``alpha = 7``, mean ``0.125``). ``Beta(alpha, 1)`` is drawn by inverse CDF, ``U ** (1 / alpha)``, so it
    follows ``generator``. ``logit_normal=(mean, std)`` draws ``sigmoid(N(mean, std))``, which spends more training on
    intermediate times, as Stable Diffusion 3 does (Esser et al., 2024, https://arxiv.org/abs/2403.03206).
    """
    if logit_normal is not None:
        if alpha is not None:
            raise ValueError("Pass alpha or logit_normal, not both")
        mean, std = logit_normal
        return torch.sigmoid(mean + std * torch.randn(shape, device=device, dtype=dtype, generator=generator))
    u = torch.rand(shape, device=device, dtype=dtype, generator=generator)
    if alpha is None:
        return u
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    return 1 - u ** (1 / alpha)


def euler_sample(velocity: Callable[[Tensor, Tensor], Tensor], noise: Tensor, steps: int) -> Tensor:
    """Integrate ``dx/dt = velocity(x, t)`` from ``t = 0`` to ``1`` in ``steps`` Euler steps; ``t`` is ``(B,)``."""
    if steps < 1:
        raise ValueError("steps must be positive")
    x = noise
    for i in range(steps):
        x = x + velocity(x, x.new_full((x.size(0),), i / steps)) / steps
    return x
