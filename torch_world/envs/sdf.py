"""Anti-aliased 2D drawing with signed distance functions, in unit coordinates with y pointing down."""

import math

import torch
from torch import Tensor


def grid(size: int, device=None, center=(0.5, 0.5), half: float = 0.5) -> tuple[Tensor, Tensor]:
    """Pixel-center coordinates ``(size, size)`` of a square window, or ``(B, size, size)`` for batched windows."""
    g = (torch.arange(size, device=device) + 0.5) / size * 2 - 1
    y, x = torch.meshgrid(g, g, indexing="ij")
    if isinstance(center, Tensor):
        return center[:, 0, None, None] + half * x, center[:, 1, None, None] + half * y
    return center[0] + half * x, center[1] + half * y


def paint(image: Tensor, sdf: Tensor, color, pixels_per_unit: float, opacity: float = 1.0) -> Tensor:
    """Composite ``color`` (RGB, or a ``(..., 3, H, W)`` color field) where ``sdf < 0``."""
    alpha = opacity * (0.5 - sdf * pixels_per_unit).clamp(0, 1).unsqueeze(-3)
    color = torch.as_tensor(color, device=image.device)
    return image + alpha * ((color[:, None, None] if color.dim() == 1 else color) - image)


def shade(t: Tensor, start, end) -> Tensor:
    """Color field blending ``start`` into ``end`` as ``t`` goes from 0 to 1."""
    t = t.clamp(0, 1).unsqueeze(-3)
    start, end = (torch.as_tensor(c, device=t.device)[:, None, None] for c in (start, end))
    return start + t * (end - start)


def fill(color, x: Tensor) -> Tensor:
    return torch.as_tensor(color, device=x.device)[:, None, None].expand(3, *x.shape[-2:])


def union(*sdfs: Tensor) -> Tensor:
    return torch.stack(torch.broadcast_tensors(*sdfs)).amin(0)


def circle(px, py, center, r) -> Tensor:
    return ((px - center[0]).square() + (py - center[1]).square()).sqrt() - r


def ellipse(px, py, center, radii) -> Tensor:
    d = (((px - center[0]) / radii[0]).square() + ((py - center[1]) / radii[1]).square()).sqrt()
    return (d - 1) * min(radii)


def box(px, py, center, half, r: float = 0.0) -> Tensor:
    """Rectangle with half sizes ``half`` and corner radius ``r``."""
    qx, qy = (px - center[0]).abs() - half[0] + r, (py - center[1]).abs() - half[1] + r
    outside = (qx.clamp(min=0).square() + qy.clamp(min=0).square()).sqrt()
    return outside + torch.maximum(qx, qy).clamp(max=0) - r


def capsule(px, py, a, b, r) -> Tensor:
    dx, dy = b[0] - a[0], b[1] - a[1]
    h = (((px - a[0]) * dx + (py - a[1]) * dy) / (dx * dx + dy * dy + 1e-9)).clamp(0, 1)
    return ((px - a[0] - h * dx).square() + (py - a[1] - h * dy).square()).sqrt() - r


def polygon(px, py, points) -> Tensor:
    """Signed distance bound for a convex polygon."""
    cx, cy = sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)
    edges = []
    for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]):
        nx, ny = y1 - y0, x0 - x1
        norm = -math.copysign(math.hypot(nx, ny), (cx - x0) * nx + (cy - y0) * ny)
        edges.append(((px - x0) * nx + (py - y0) * ny) / norm)
    return torch.stack(torch.broadcast_tensors(*edges)).amax(0)
