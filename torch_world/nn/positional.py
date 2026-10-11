import math

import torch
from torch import Tensor


def sincos_1d(dim: int, pos: Tensor) -> Tensor:
    omega = 1.0 / 10000 ** (torch.arange(dim // 2, dtype=torch.float64) / (dim / 2))
    out = pos.double()[..., None] * omega
    return torch.cat([out.sin(), out.cos()], dim=-1).float()


def sincos_2d(dim: int, grid: tuple[int, int]) -> Tensor:
    """Fixed 2D sin-cos embedding of shape ``(H * W, dim)``, as in V-JEPA."""
    rows, cols = torch.meshgrid(torch.arange(grid[0]), torch.arange(grid[1]), indexing="ij")
    return torch.cat([sincos_1d(dim // 2, rows.flatten()), sincos_1d(dim // 2, cols.flatten())], dim=-1)


def sincos_3d(dim: int, grid: tuple[int, int, int]) -> Tensor:
    """Fixed 3D sin-cos embedding of shape ``(T * H * W, dim)`` with uniform power per axis, as in V-JEPA."""
    sub = 2 * math.ceil(dim / 6)
    coords = torch.meshgrid(*(torch.arange(n) for n in grid), indexing="ij")
    return torch.cat([sincos_1d(sub, c.flatten()) for c in coords], dim=-1)[:, :dim]


def sincos(dim: int, grid: tuple[int, ...]) -> Tensor:
    if len(grid) == 1:
        return sincos_1d(dim, torch.arange(grid[0]))
    return sincos_2d(dim, grid) if len(grid) == 2 else sincos_3d(dim, grid)


def grid_coords(grid: tuple[int, ...]) -> Tensor:
    """Integer coordinates of every cell of ``grid`` in row-major order, shape ``(prod(grid), len(grid))``."""
    coords = torch.meshgrid(*(torch.arange(n) for n in grid), indexing="ij")
    return torch.stack([c.flatten() for c in coords], dim=-1).float()


def rope(x: Tensor, pos: Tensor) -> Tensor:
    """Rotary embedding over ``pos.size(-1)`` axes, each rotating an equal even slice of the last dim of ``x``.

    ``x`` is ``(..., N, D)`` and ``pos`` is ``(N, A)`` or broadcastable to ``(..., N, A)``.
    Dimensions left after ``A`` equal slices are not rotated.
    """
    axes = pos.size(-1)
    d = 2 * (x.size(-1) // axes // 2)
    omega = 1.0 / 10000 ** (torch.arange(d // 2, device=x.device, dtype=torch.float32) / (d / 2))
    parts = []
    for i in range(axes):
        xi = x[..., i * d : (i + 1) * d]
        angle = (pos[..., i, None].float() * omega).repeat_interleave(2, dim=-1).to(x.dtype)
        x1, x2 = xi.unflatten(-1, (-1, 2)).unbind(-1)
        rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
        parts.append(xi * angle.cos() + rotated * angle.sin())
    return torch.cat(parts + [x[..., axes * d :]], dim=-1)
