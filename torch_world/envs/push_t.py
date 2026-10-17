"""Push-T in MuJoCo: push a T-shaped block to a target pose with a circular pusher; requires ``mujoco``."""

import functools
import math

import numpy as np
import torch
from torch import Tensor

from torch_world.envs.legacy import LegacyEnvironment
from torch_world.envs.mujoco import MujocoEnv, add_ghost, import_mujoco, look_at, studio

# The reference task is a 512 px square drawn by pymunk; one pixel is one millimeter here, with the origin at its
# center and y pointing up (pixel rows point down).
PIXELS = 512
VIEW = PIXELS / 1000  # side of the top camera view, in meters
WALLS = (5 - PIXELS / 2) / 1000, (506 - PIXELS / 2) / 1000, 0.002  # wall center lines and half thickness
BAR, STEM = (0.06, 0.015), (0.015, 0.045)  # half sizes of the T's bar and stem
BAR_Y, STEM_Y = -0.015, -0.075  # their centers below the block's origin, the middle of the bar's top edge
COG, INERTIA = (0.0, -0.045), 0.003  # the reference block's center of gravity and moment (kg m^2, for 1 kg)
PUSHER, HEIGHT = 0.015, 0.012  # pusher radius, half height of the T
TARGET = (0.0, 0.0, -math.pi / 4)  # the green target T drawn on the table, as in the reference renders
T_COLOR, PUSHER_COLOR = (0.467, 0.533, 0.6, 1.0), (0.255, 0.412, 0.882, 1.0)  # LightSlateGray, RoyalBlue
TARGET_COLOR, WALL = (0.565, 0.933, 0.565, 1.0), (0.83, 0.83, 0.83, 1.0)  # LightGreen, LightGray
TICK = 0.01  # the reference's physics step, at which the pusher's PD law runs and the block's velocity is reset


class PushT(MujocoEnv, LegacyEnvironment):
    """Push-T: a circular pusher moves on a table and pushes a T-shaped block toward a target pose, seen from above.

    The planar task of Diffusion Policy, in the version DINO-WM and LeWM train and evaluate on
    (``stable_worldmodel``'s ``swm/PushT-v1``, observed at ``size=224``), simulated in MuJoCo. The scene, the T, the pusher and the walls have
    the reference sizes (one reference pixel is one millimeter), and the dynamics follow it: actions ``(dx, dy)`` in
    ``[-1, 1]^2`` place the pusher's target ``max_step`` meters per unit from its current position, a critically
    damped PD law moves the kinematic pusher toward it every 10 ms (passing through the walls), contacts are
    frictionless, and the block stops as soon as it is no longer pushed (its velocity is reset every 10 ms, the
    reference's zero damping). Each step lasts ``control_dt`` seconds.

    The state is ``(pusher x, y, block x, y, angle, pusher vx, vy)`` in meters, radians and meters per second, with
    the arena centered at the origin and the block's position at the middle of the top edge of the T's bar, as in
    the reference; ``from_reference`` and ``to_reference`` convert reference states (pixels, y down). Resets take
    7-dimensional states, or 5-dimensional ones with the pusher at rest. Success is the reference's: pusher and
    block positions together within ``radius`` of the goal's, and the block within ``max_angle`` of its angle.
    A green T marks the fixed target pose in every image, as in the reference renders; ``target=False`` hides it.

    Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023.
    Paper: https://arxiv.org/abs/2303.04137
    Code: https://github.com/real-stanford/diffusion_policy
    """

    env_id = "PushT-v0"
    action_dim = 2
    state_dim = 7
    radius = 0.02
    max_angle = math.pi / 9
    top_size = VIEW
    kp, kv = 100.0, 20.0  # the reference's PD gains on the pusher

    def __init__(
        self,
        batch_size: int = 1,
        size: int = 96,
        control_dt: float = 0.1,
        max_step: float = 0.1,
        device=None,
        *,
        target: bool = True,
        quality="high",
        render_size=None,
        max_episode_steps=100,
        observation="rgb",
    ):
        super().__init__(_build(target), batch_size, size, "top", device)
        self.max_step = max_step
        self.ticks = max(1, round(control_dt / TICK))
        self.substeps = max(1, round(TICK / self.model.opt.timestep))
        self.pusher, self.velocity = np.zeros((batch_size, 2)), np.zeros((batch_size, 2))
        self.plans = np.zeros((batch_size, 3, 2))
        self.phase = np.full(batch_size, 3)
        self._sync()
        self._configure_benchmark(
            quality=quality,
            render_size=render_size,
            max_episode_steps=max_episode_steps,
            observation=observation,
            control_dt=control_dt,
            max_step=max_step,
        )

    def reset(self, state: Tensor | None = None, generator: torch.Generator | None = None) -> Tensor:
        """A random block pose and pusher position from the reference's ranges, or the given ``state``."""
        if state is None:
            u = torch.rand(self.batch_size, 5, generator=generator).double()
            pixels = torch.cat([50 + 400 * u[:, :2], 100 + 300 * u[:, 2:4], 4 * math.pi * (u[:, 4:] - 0.5)], 1)
            state = self.from_reference(pixels)
        s = state.double().cpu().numpy()
        if s.shape[1] == 5:
            s = np.concatenate([s, np.zeros((len(s), 2))], 1)
        for d, row in zip(self.data, s):
            self._place(d, row)
        self.pusher, self.velocity = s[:, :2].copy(), s[:, 5:7].copy()
        self._sync()
        return self.render()

    def step(self, action: Tensor) -> Tensor:
        a = action.clamp(-1, 1).double().cpu().numpy()
        targets = self.pusher + self.max_step * a
        for _ in range(self.ticks):
            self.velocity = self.velocity + TICK * (self.kp * (targets - self.pusher) - self.kv * self.velocity)
            for b, d in enumerate(self.data):
                d.qpos[:2], d.qvel[:2], d.qvel[2:5] = self.pusher[b], self.velocity[b], 0.0
                self.mujoco.mj_step(self.model, d, self.substeps)
            self.pusher = self.pusher + TICK * self.velocity
        for b, d in enumerate(self.data):  # the pusher is kinematic: contacts never deflect it
            d.qpos[:2], d.qvel[:2] = self.pusher[b], self.velocity[b]
            self.mujoco.mj_kinematics(self.model, d)
        self._sync()
        return self.render()

    def _place(self, d, state: np.ndarray):
        self.mujoco.mj_resetData(self.model, d)
        d.qpos[:5] = state[:5]
        if len(state) > 5:
            d.qvel[:2] = state[5:7]
        self.mujoco.mj_forward(self.model, d)

    def _state(self, b: int, d) -> list[float]:
        return [*d.qpos[:5].tolist(), *d.qvel[:2].tolist()]

    def _ghost(self, goal: np.ndarray):
        return [goal[2], goal[3], HEIGHT], [math.cos(goal[4] / 2), 0, 0, math.sin(goal[4] / 2)]

    @staticmethod
    def from_reference(states: Tensor) -> Tensor:
        """States of this env from reference states ``(..., 5 or 7)`` in pixels, radians and pixels per second, with
        y pointing down; ``to_reference`` is the inverse."""
        shift, scale = _reference(states)
        return (states - shift) / scale

    @staticmethod
    def to_reference(states: Tensor) -> Tensor:
        shift, scale = _reference(states)
        return states * scale + shift

    # scoring

    def position(self, states: Tensor) -> Tensor:
        """The block's position in the unit square of the top camera (x right, y down)."""
        return torch.stack([0.5 + states[..., 2] / self.top_size, 0.5 - states[..., 3] / self.top_size], dim=-1)

    def distance(self, states: Tensor, goal: Tensor) -> Tensor:
        """Distance in meters between the block and its goal position, for states ``(B, ..., D)`` and goals ``(B, D)``."""
        goal = goal.view(len(goal), *[1] * (states.dim() - 2), -1)
        return (states[..., 2:4] - goal[..., 2:4]).norm(dim=-1)

    def angle(self, states: Tensor, goal: Tensor) -> Tensor:
        """Absolute angle in radians between the block and its goal orientation."""
        goal = goal.view(len(goal), *[1] * (states.dim() - 2), -1)
        return torch.remainder(states[..., 4] - goal[..., 4] + math.pi, 2 * math.pi).sub(math.pi).abs()

    def success(self, states: Tensor, goal: Tensor) -> Tensor:
        """The reference test: the distance between the stacked pusher and block positions and the goal's is below
        ``radius``, and the block's angle is within ``max_angle`` of the goal's."""
        g = goal.view(len(goal), *[1] * (states.dim() - 2), -1)
        near = (states[..., :4] - g[..., :4]).norm(dim=-1) < self.radius
        return near & (self.angle(states, goal) < self.max_angle)

    def coverage(self, states: Tensor, goal: Tensor) -> Tensor:
        """Fraction of the goal T's area covered by the block, for states ``(B, ..., D)`` and goals ``(B, D)``: the
        score of Diffusion Policy's Push-T, which counts coverage above 95% as success. Measured on a 2 mm grid."""
        points = _t_points(states.device, states.dtype)
        goal = goal.view(len(goal), *[1] * (states.dim() - 2), -1).expand(*states.shape[:-1], -1)
        world = _rotate(points, goal[..., 4:5]) + goal[..., None, 2:4]
        local = _rotate(world - states[..., None, 2:4], -states[..., 4:5])
        return _inside_t(local).float().mean(-1)

    # data

    def explore(self, generator: torch.Generator | None = None) -> Tensor:
        """Actions ``(B, 2)`` of a pushing policy: approach a random point of the block's edge from outside, then
        push through it. Each push plans an approach, a contact and an end point in the block's frame."""
        s = self.state.double().cpu().numpy()
        new = self.phase >= 3
        if new.any():
            u = torch.rand(self.batch_size, 3, generator=generator).double().numpy()
            point, normal = _edge(u[:, 0])
            depth = 0.03 + 0.05 * u[:, 1]
            local = np.stack([point + 0.035 * normal, point, point - depth[:, None] * normal], 1)
            c, sn = np.cos(s[:, 4])[:, None], np.sin(s[:, 4])[:, None]
            world = np.stack([c * local[..., 0] - sn * local[..., 1], sn * local[..., 0] + c * local[..., 1]], -1)
            self.plans = np.where(new[:, None, None], world + s[:, None, 2:4], self.plans)
            self.phase = np.where(new, 0, self.phase)
        waypoint = self.plans[np.arange(self.batch_size), self.phase]
        move = (waypoint - s[:, :2]) / self.max_step
        arrived = np.linalg.norm(waypoint - s[:, :2], axis=1) < 0.01
        self.phase = self.phase + arrived
        return torch.from_numpy(np.clip(move, -1, 1)).float().to(self.state.device)

    def tasks(self, steps: int = 30, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
        """Random starts, and goals where the pushing policy has moved the block for ``steps`` steps, or longer (up to
        ``4 * steps``) until the start no longer counts as reaching the goal."""
        self.reset(generator=generator)
        start = self.state.clone()
        self.phase = np.full(self.batch_size, 3)
        goal, done = start.clone(), torch.zeros(self.batch_size, dtype=torch.bool, device=start.device)
        for t in range(4 * steps):
            self.step(self.explore(generator))
            if t + 1 >= steps:
                take = ~done & ~self.success(start, self.state) | (t + 1 == 4 * steps)
                goal[take & ~done], done = self.state[take & ~done], done | take
                if done.all():
                    break
        self.reset(start)
        return start, goal

    def collect(self, num_steps: int, noise: float = 0.3, generator: torch.Generator | None = None) -> dict:
        """Episodes of the noisy pushing policy: ``pixels`` ``(B, T, 3, size, size)``, ``actions`` ``(B, T, 2)`` and
        ``states`` ``(B, T, 7)``; ``actions[:, t]`` is applied at ``states[:, t]``."""
        pixels, states, actions = [self.reset(generator=generator)], [self.state], []
        self.phase = np.full(self.batch_size, 3)
        for _ in range(num_steps - 1):
            n = torch.randn(self.batch_size, 2, generator=generator).to(self.state)
            action = (self.explore(generator) + noise * n).clamp(-1, 1)
            actions.append(action)
            pixels.append(self.step(action))
            states.append(self.state)
        actions.append(torch.zeros_like(actions[-1]))
        return {"pixels": torch.stack(pixels, 1), "actions": torch.stack(actions, 1), "states": torch.stack(states, 1)}


def _edge(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Points on the T's outline and their outward normals, at fractions ``u`` of its perimeter (block frame)."""
    (bx, by), (sx, sy) = BAR, STEM
    top, under, foot = BAR_Y + by, BAR_Y - by, STEM_Y - sy
    sides = [  # start, end, outward normal
        ((-bx, top), (bx, top), (0, 1)),
        ((bx, top), (bx, under), (1, 0)),
        ((bx, under), (sx, under), (0, -1)),
        ((sx, under), (sx, foot), (1, 0)),
        ((sx, foot), (-sx, foot), (0, -1)),
        ((-sx, foot), (-sx, under), (-1, 0)),
        ((-sx, under), (-bx, under), (0, -1)),
        ((-bx, under), (-bx, top), (-1, 0)),
    ]
    lengths = np.array([math.dist(a, b) for a, b, _ in sides])
    edges = np.cumsum(lengths) / lengths.sum()
    k = np.searchsorted(edges, u.clip(0, 1 - 1e-9), side="right")
    t = (u - np.concatenate([[0], edges])[k]) * lengths.sum() / lengths[k]
    a, b = np.array([s[0] for s in sides])[k], np.array([s[1] for s in sides])[k]
    return a + t[:, None] * (b - a), np.array([s[2] for s in sides], dtype=float)[k]


def _reference(states: Tensor) -> tuple[Tensor, Tensor]:
    """Shift and signed scale from this env's state coordinates to the reference's: ``reference = state * scale +
    shift``, for the 5 or 7 leading state dimensions of ``states``."""
    d = states.shape[-1]
    shift = torch.tensor([PIXELS / 2] * 4 + [0.0] * 3, dtype=states.dtype, device=states.device)[:d]
    scale = torch.tensor([1e3, -1e3, 1e3, -1e3, -1.0, 1e3, -1e3], dtype=states.dtype, device=states.device)[:d]
    return shift, scale


def _rotate(points: Tensor, angle: Tensor) -> Tensor:
    x, y, c, s = points[..., 0], points[..., 1], angle.cos(), angle.sin()
    return torch.stack([c * x - s * y, s * x + c * y], -1)


def _inside_t(p: Tensor) -> Tensor:
    x, y = p[..., 0].abs(), p[..., 1]
    bar = (x <= BAR[0]) & ((y - BAR_Y).abs() <= BAR[1])
    stem = (x <= STEM[0]) & ((y - STEM_Y).abs() <= STEM[1])
    return bar | stem


@functools.cache
def _t_points(device, dtype, spacing: float = 0.002) -> Tensor:
    """Cell centers of a ``spacing`` grid over the bar and the stem, each at least half a cell inside its edges."""

    def cells(half: tuple[float, float], center_y: float) -> Tensor:
        xs = torch.arange(-half[0] + spacing / 2, half[0], spacing, dtype=torch.float64)
        ys = torch.arange(-half[1] + spacing / 2, half[1], spacing, dtype=torch.float64) + center_y
        return torch.cartesian_prod(xs, ys)

    return torch.cat([cells(BAR, BAR_Y), cells(STEM, STEM_Y)]).to(device, dtype)


def _build(target: bool = True):
    mujoco = import_mujoco()
    spec = studio(scale=0.6, table_half=0.45)
    spec.option.timestep = 0.002
    spec.option.gravity = [0, 0, 0]
    world = spec.worldbody
    lo, hi, half = WALLS
    mid, length = (lo + hi) / 2, (hi - lo) / 2 + half
    for x, y, hx, hy in [(mid, lo, length, half), (mid, hi, length, half), (lo, mid, half, length),
                         (hi, mid, half, length)]:  # fmt: skip
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, 0.01], pos=[x, y, 0.01], rgba=list(WALL),
                       contype=4, conaffinity=4)  # fmt: skip
    t = [
        {"type": mujoco.mjtGeom.mjGEOM_BOX, "size": [*BAR, HEIGHT], "pos": [0, BAR_Y, 0]},
        {"type": mujoco.mjtGeom.mjGEOM_BOX, "size": [*STEM, HEIGHT], "pos": [0, STEM_Y, 0]},
    ]
    if target:  # flat, drawn just above the table, never touched
        x, y, a = TARGET
        marker = world.add_body(name="target", pos=[x, y, 0.0005], quat=[math.cos(a / 2), 0, 0, math.sin(a / 2)])
        for g in t:
            marker.add_geom(type=g["type"], size=[*g["size"][:2], 0.0005], pos=g["pos"], rgba=list(TARGET_COLOR),
                            contype=0, conaffinity=0)  # fmt: skip

    # contacts are frictionless (condim 1) and stiff, so the block moves with the pusher like the reference's
    contact = {"condim": 1, "solref": [0.004, 1], "solimp": [0.95, 0.99, 0.001, 0.5, 2]}
    pusher = world.add_body(name="pusher", pos=[0, 0, HEIGHT], mass=100, inertia=[1, 1, 1], explicitinertial=True)
    for axis, name in (([1, 0, 0], "pusher_x"), ([0, 1, 0], "pusher_y")):
        pusher.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis)
    pusher.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[PUSHER, HEIGHT, 0], rgba=list(PUSHER_COLOR),
                    contype=2, conaffinity=2, **contact)  # fmt: skip

    block = world.add_body(name="block", pos=[0, 0, HEIGHT], mass=1, ipos=[*COG, 0],
                           inertia=[INERTIA, INERTIA, INERTIA], explicitinertial=True)  # fmt: skip
    for axis, name in (([1, 0, 0], "block_x"), ([0, 1, 0], "block_y")):
        block.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis)
    block.add_joint(name="block_angle", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1])
    for g in t:
        block.add_geom(**g, rgba=list(T_COLOR), contype=6, conaffinity=6, **contact)
    add_ghost(spec, t)

    eye = [0, 0, 1.2]
    world.add_camera(name="top", pos=eye, quat=look_at(eye, [0, 0, 0], up=(0, 1, 0)), fovy=VIEW,
                     proj=mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC)  # fmt: skip
    world.add_camera(name="front", pos=[0, -0.75, 0.6], quat=look_at([0, -0.75, 0.6], [0, -0.02, 0]), fovy=40)
    return spec.compile()
