import math

import torch
from torch import Tensor

from torch_world.envs.point_mass import PointMass
from torch_world.envs.sdf import capsule, circle, ellipse, fill, grid, paint, polygon, shade, union

SKY, HORIZON, SUN = (0.33, 0.6, 0.88), (0.98, 0.86, 0.7), (1.0, 0.95, 0.74)
FAR_HILLS, NEAR_HILLS, CLOUD = (0.56, 0.68, 0.64), (0.3, 0.5, 0.36), (1.0, 1.0, 1.0)
FAR_PALM, NEAR_PALM = (0.43, 0.57, 0.52), (0.2, 0.37, 0.27)
CLOUDS = [(0.2, 0.2, 1.0), (0.55, 0.3, 0.8), (0.88, 0.47, 0.65)]
PALMS = [(0.1, 0.84, 0.3, NEAR_PALM), (0.66, 0.8, 0.17, FAR_PALM), (0.93, 0.86, 0.24, NEAR_PALM)]

HIGHLIGHT, COBALT, SHADOW = (0.33, 0.48, 0.88), (0.17, 0.28, 0.68), (0.07, 0.12, 0.37)
FLIGHT, FEATHER_EDGE = (0.1, 0.17, 0.48), (0.05, 0.08, 0.25)
BEAK, BEAK_SHINE, YELLOW, PUPIL = (0.13, 0.13, 0.15), (0.3, 0.3, 0.33), (0.99, 0.8, 0.16), (0.04, 0.04, 0.06)
TAIL = [
    ((-0.05, -0.02), (-0.05, 0.012), (-0.2, 0.01), (-0.232, 0.0), (-0.215, -0.009)),
    ((-0.05, -0.004), (-0.05, 0.02), (-0.185, 0.027), (-0.205, 0.02), (-0.19, 0.012)),
]
HOOK = ((0.098, -0.04), (0.124, -0.024), (0.124, 0.003), (0.113, 0.012), (0.096, -0.004))
# primary and secondary feathers: (position along the arm, angle from the arm, length, radius)
FEATHERS = [
    (1.0, 0.05, 0.1, 0.008),
    (1.0, -0.13, 0.1, 0.008),
    (1.0, -0.31, 0.093, 0.0085),
    (0.95, -0.5, 0.083, 0.009),
    (0.75, -0.72, 0.07, 0.01),
    (0.5, -0.92, 0.058, 0.011),
    (0.25, -1.08, 0.048, 0.012),
]


class Macaw(PointMass):
    """A Hyacinth macaw flying over the Pantanal, on ``size x size`` RGB images: the same dynamics as ``PointMass``.

    The state is the bird's position, kept in ``[low, high]^2``. Its wings beat a quarter cycle per step, so a single
    frame does not tell the phase but a short history does. ``render(state)`` without a phase draws the bird gliding,
    as used for goal images. Rendering is resolution-free: a ``Macaw(size=192)`` draws the same states sharper.
    """

    env_id = "Macaw-v0"
    low, high = 0.15, 0.85

    def __init__(
        self,
        batch_size: int = 1,
        size: int = 64,
        max_step: float = 0.1,
        scale: float = 1.0,
        device=None,
        *,
        quality="high",
        render_size=None,
        max_episode_steps=100,
        observation="rgb",
    ):
        super().__init__(batch_size, size, max_step, device=device)
        self.scale = scale
        self.phase = torch.zeros(batch_size, device=device)
        self.landscape = landscape(size, device)
        self._configure_benchmark(
            quality=quality,
            render_size=render_size,
            max_episode_steps=max_episode_steps,
            observation=observation,
            max_step=max_step,
            scale=scale,
        )

    def reset(self, state: Tensor | None = None, generator: torch.Generator | None = None) -> Tensor:
        self.phase = torch.zeros(self.batch_size, device=self.device)
        return super().reset(state, generator)

    def step(self, action: Tensor) -> Tensor:
        self.phase = self.phase + math.pi / 2
        return super().step(action)

    def render(self, state: Tensor | None = None, phase: Tensor | None = None) -> Tensor:
        """Images ``(B, 3, size, size)`` in ``[0, 1]``; ``phase`` ``(B,)`` sets the wing beat."""
        if state is None:
            state, phase = self.state, self.phase if phase is None else phase
        phase = torch.zeros(len(state), device=state.device) if phase is None else phase
        x, y = grid(self.size, state.device)
        dx, dy = (x - state[:, 0, None, None]) / self.scale, (y - state[:, 1, None, None]) / self.scale
        pitch = -0.1
        px, py = dx * math.cos(pitch) + dy * math.sin(pitch), -dx * math.sin(pitch) + dy * math.cos(pitch)
        ppu = self.size * self.scale
        image = self.landscape.to(state.device).expand(len(state), -1, -1, -1)
        lift = phase.sin()[:, None, None]

        def draw(sdf, color):
            nonlocal image
            image = paint(image, sdf, color, ppu)

        draw(_wing(px, py, lift, (0.026, -0.024), 0.85), FEATHER_EDGE)
        for k, tail in enumerate(TAIL[::-1]):
            draw(polygon(px, py, tail), shade((py + 0.01) / 0.03, COBALT, SHADOW) if k else FLIGHT)
        body = union(
            ellipse(px, py, (0.0, 0.0), (0.078, 0.031)),
            capsule(px, py, (0.03, -0.006), (0.068, -0.02), 0.027),
            circle(px, py, (0.072, -0.022), 0.031),
        )
        draw(body, shade((py + 0.035) / 0.065, HIGHLIGHT, SHADOW))
        draw(union(circle(px, py, (0.103, -0.02), 0.021), polygon(px, py, HOOK)), BEAK)
        draw(ellipse(px, py, (0.104, -0.031), (0.01, 0.004)), BEAK_SHINE)
        draw(ellipse(px, py, (0.099, 0.005), (0.016, 0.009)), BEAK)
        draw(ellipse(px, py, (0.089, 0.008), (0.012, 0.0045)), YELLOW)
        draw(circle(px, py, (0.079, -0.028), 0.0095), YELLOW)
        draw(circle(px, py, (0.08, -0.028), 0.005), PUPIL)
        wing = _wing(px, py, lift, (0.018, -0.018), 1.0, edge=0.003)
        draw(wing, FEATHER_EDGE)
        draw(_wing(px, py, lift, (0.018, -0.018), 1.0), shade((py + 0.1) / 0.2, FLIGHT, SHADOW))
        draw(capsule(px, py, (0.018, -0.018), _wrist((0.018, -0.018), lift, 1.0), 0.022), COBALT)
        return image


def _angle(lift: Tensor) -> Tensor:
    return 0.3 + 1.05 * lift


def _direction(lift: Tensor) -> tuple[Tensor, Tensor]:
    angle = _angle(lift)
    return -angle.cos(), -angle.sin()


def _wrist(shoulder, lift: Tensor, size: float):
    ux, uy = _direction(lift)
    return shoulder[0] + 0.062 * size * ux, shoulder[1] + 0.062 * size * uy


def _wing(px, py, lift: Tensor, shoulder, size: float, edge: float = 0.0) -> Tensor:
    """Union of feathers fanned along the arm from ``shoulder`` to the wrist, rotated by the wing beat."""
    angle = _angle(lift)
    wrist = _wrist(shoulder, lift, size)
    feathers = []
    for at, offset, length, r in FEATHERS:
        base = (shoulder[0] + at * (wrist[0] - shoulder[0]), shoulder[1] + at * (wrist[1] - shoulder[1]))
        a = angle + offset
        tip = (base[0] - size * length * a.cos(), base[1] - size * length * a.sin())
        feathers.append(capsule(px, py, base, tip, size * r + edge))
    return union(*feathers)


def landscape(size: int, device=None) -> Tensor:
    x, y = grid(size, device)
    t = (y.clamp(0, 0.82) / 0.82)[None]
    image = fill(SKY, x) * (1 - t) + fill(HORIZON, x) * t
    glow = torch.exp(-((x - 0.8).square() + (y - 0.17).square()) / 0.03)
    image = image + 0.3 * glow * (fill(SUN, x) - image)
    image = paint(image, circle(x, y, (0.8, 0.17), 0.065), SUN, size)
    puffs = [(-0.07, 0.01, 0.035), (-0.025, -0.012, 0.045), (0.025, -0.005, 0.04), (0.07, 0.012, 0.03)]
    for cx, cy, s in CLOUDS:
        cloud = union(*(circle(x, y, (cx + s * dx, cy + s * dy), s * r) for dx, dy, r in puffs))
        cloud = union(cloud, ellipse(x, y, (cx, cy + 0.015 * s), (0.09 * s, 0.022 * s)))
        image = paint(image, cloud, shade((y - cy + 0.04 * s) / (0.08 * s), CLOUD, (0.9, 0.9, 0.95)), size, 0.9)
    far = y - (0.8 + 0.035 * torch.sin(7 * x + 1) + 0.015 * torch.sin(19 * x))
    image = paint(image, -far, FAR_HILLS, size)
    for px, base, height, color in PALMS:
        image = paint(image, _palm(x, y, px, base, height), color, size)
    near = y - (0.88 + 0.03 * torch.sin(5 * x + 3) + 0.01 * torch.sin(23 * x + 1))
    return paint(image, -near, NEAR_HILLS, size)


def _palm(x, y, at: float, base: float, height: float) -> Tensor:
    """A buriti palm: a slightly leaning trunk and a crown of drooping fronds."""
    crown = (at + 0.03 * height, base - height)
    parts = [capsule(x, y, (at, base), crown, 0.004 + 0.012 * height)]
    reach, width = 0.38 * height, 0.012 * height + 0.003
    for angle in (-3.0, -2.6, -2.2, -1.85, -1.55, -1.25, -0.9, -0.5, -0.1):
        point, heading = crown, angle
        for k in range(3):
            heading = heading + (0.35 if math.cos(heading) > 0 else -0.35) * (k > 0)
            nxt = (point[0] + reach / 3 * math.cos(heading), point[1] + reach / 3 * math.sin(heading))
            parts.append(capsule(x, y, point, nxt, width * (1 - 0.3 * k)))
            point = nxt
    return union(*parts)
