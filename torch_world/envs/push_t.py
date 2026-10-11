"""Push-T in MuJoCo: push a T-shaped block to a target pose with a circular pusher; requires ``mujoco``."""

import functools
import math

import numpy as np
import torch
from torch import Tensor

from torch_world.envs.mujoco import MujocoEnv, add_ghost, import_mujoco, look_at, studio

ARENA = 0.25  # half side of the square arena, in meters
VIEW = 2 * ARENA + 0.04  # side of the top camera view
BAR, STEM = (0.06, 0.015), (0.015, 0.045)  # half sizes of the T's bar and stem
STEM_OFFSET = BAR[1] + STEM[1]  # stem center below the bar center
PUSHER, HEIGHT = 0.015, 0.012  # pusher radius, half height of the T
T_COLOR, PUSHER_COLOR, WALL = (0.42, 0.48, 0.56, 1.0), (0.16, 0.47, 0.84, 1.0), (0.78, 0.76, 0.72, 1.0)


class PushT(MujocoEnv):
    """Push-T: a circular pusher moves on a table and pushes a T-shaped block toward a target pose, seen from above.

    The planar task of Diffusion Policy, used by DINO-WM and LeWM, simulated in MuJoCo with contact and table
    friction, in a light studio scene. Actions ``(dx, dy)`` in ``[-1, 1]^2`` move the pusher's target by up to
    ``max_step`` meters; each step lasts ``control_dt`` seconds. The state is ``(pusher x, y, block x, y, angle)``
    in meters and radians, with the arena centered at the origin. Success means the block is within ``radius`` of
    its goal position and ``max_angle`` of its goal angle.

    Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023.
    Paper: https://arxiv.org/abs/2303.04137
    Code: https://github.com/real-stanford/diffusion_policy
    """

    action_dim = 2
    state_dim = 5
    radius = 0.025
    max_angle = 0.25
    top_size = VIEW

    def __init__(
        self,
        batch_size: int = 1,
        size: int = 96,
        control_dt: float = 0.1,
        max_step: float = 0.025,
        device=None,
    ):
        super().__init__(_build(), batch_size, size, "top", device)
        self.max_step = max_step
        self.substeps = max(1, round(control_dt / self.model.opt.timestep))
        self.targets = np.zeros((batch_size, 2))
        self.plans = np.zeros((batch_size, 3, 2))
        self.phase = np.full(batch_size, 3)
        self._sync()

    def reset(self, state: Tensor | None = None, generator: torch.Generator | None = None) -> Tensor:
        """A random block pose and a pusher away from the block, or the given ``state``."""
        if state is None:
            u = torch.rand(self.batch_size, 5, generator=generator).double().numpy()
            block = np.stack([(u[:, 0] - 0.5) * ARENA, (u[:, 1] - 0.5) * ARENA, (u[:, 2] - 0.5) * 2 * math.pi], 1)
            angle = u[:, 3] * 2 * math.pi
            pusher = block[:, :2] + (0.12 + 0.05 * u[:, 4, None]) * np.stack([np.cos(angle), np.sin(angle)], 1)
            state = torch.from_numpy(np.concatenate([np.clip(pusher, -0.9 * ARENA, 0.9 * ARENA), block], 1))
        s = state.double().cpu().numpy()
        for d, row in zip(self.data, s):
            self._place(d, row)
        self.targets = s[:, :2].copy()
        self._sync()
        return self.render()

    def step(self, action: Tensor) -> Tensor:
        a = action.clamp(-1, 1).double().cpu().numpy()
        self.targets = np.clip(self.targets + self.max_step * a, -ARENA + PUSHER, ARENA - PUSHER)
        for b, d in enumerate(self.data):
            d.ctrl[:] = self.targets[b]
            self.mujoco.mj_step(self.model, d, self.substeps)
        self._sync()
        return self.render()

    def _place(self, d, state: np.ndarray):
        self.mujoco.mj_resetData(self.model, d)
        d.qpos[:5] = state
        d.ctrl[:] = state[:2]
        self.mujoco.mj_forward(self.model, d)

    def _state(self, b: int, d) -> list[float]:
        return d.qpos[:5].tolist()

    def _ghost(self, goal: np.ndarray):
        return [goal[2], goal[3], HEIGHT], [math.cos(goal[4] / 2), 0, 0, math.sin(goal[4] / 2)]

    # scoring

    def position(self, states: Tensor) -> Tensor:
        """The block's position in the unit square of the top camera (x right, y down)."""
        return torch.stack([0.5 + states[..., 2] / self.top_size, 0.5 - states[..., 3] / self.top_size], dim=-1)

    def distance(self, states: Tensor, goal: Tensor) -> Tensor:
        """Distance in meters between the block and its goal position, for states ``(B, ..., 5)`` and goals ``(B, 5)``."""
        goal = goal.view(len(goal), *[1] * (states.dim() - 2), -1)
        return (states[..., 2:4] - goal[..., 2:4]).norm(dim=-1)

    def angle(self, states: Tensor, goal: Tensor) -> Tensor:
        """Absolute angle in radians between the block and its goal orientation."""
        goal = goal.view(len(goal), *[1] * (states.dim() - 2), -1)
        return torch.remainder(states[..., 4] - goal[..., 4] + math.pi, 2 * math.pi).sub(math.pi).abs()

    def success(self, states: Tensor, goal: Tensor) -> Tensor:
        return (self.distance(states, goal) <= self.radius) & (self.angle(states, goal) <= self.max_angle)

    def coverage(self, states: Tensor, goal: Tensor) -> Tensor:
        """Fraction of the goal T's area covered by the block, for states ``(B, ..., 5)`` and goals ``(B, 5)``: the
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
        move = (waypoint - self.targets) / self.max_step
        arrived = np.linalg.norm(waypoint - self.targets, axis=1) < self.max_step
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
        ``states`` ``(B, T, 5)``; ``actions[:, t]`` is applied at ``states[:, t]``."""
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
    (bx, by), (sx, sy), o = BAR, STEM, STEM_OFFSET
    sides = [  # start, end, outward normal
        ((-bx, by), (bx, by), (0, 1)),
        ((bx, by), (bx, -by), (1, 0)),
        ((bx, -by), (sx, -by), (0, -1)),
        ((sx, -by), (sx, -o - sy), (1, 0)),
        ((sx, -o - sy), (-sx, -o - sy), (0, -1)),
        ((-sx, -o - sy), (-sx, -by), (-1, 0)),
        ((-sx, -by), (-bx, -by), (0, -1)),
        ((-bx, -by), (-bx, by), (-1, 0)),
    ]
    lengths = np.array([math.dist(a, b) for a, b, _ in sides])
    edges = np.cumsum(lengths) / lengths.sum()
    k = np.searchsorted(edges, u.clip(0, 1 - 1e-9), side="right")
    t = (u - np.concatenate([[0], edges])[k]) * lengths.sum() / lengths[k]
    a, b = np.array([s[0] for s in sides])[k], np.array([s[1] for s in sides])[k]
    return a + t[:, None] * (b - a), np.array([s[2] for s in sides], dtype=float)[k]


def _rotate(points: Tensor, angle: Tensor) -> Tensor:
    x, y, c, s = points[..., 0], points[..., 1], angle.cos(), angle.sin()
    return torch.stack([c * x - s * y, s * x + c * y], -1)


def _inside_t(p: Tensor) -> Tensor:
    x, y = p[..., 0].abs(), p[..., 1]
    bar = (x <= BAR[0]) & (y.abs() <= BAR[1])
    stem = (x <= STEM[0]) & ((y + STEM_OFFSET).abs() <= STEM[1])
    return bar | stem


@functools.cache
def _t_points(device, dtype, spacing: float = 0.002) -> Tensor:
    """Cell centers of a ``spacing`` grid over the bar and the stem, each at least half a cell inside its edges."""

    def cells(half: tuple[float, float], center_y: float) -> Tensor:
        xs = torch.arange(-half[0] + spacing / 2, half[0], spacing, dtype=torch.float64)
        ys = torch.arange(-half[1] + spacing / 2, half[1], spacing, dtype=torch.float64) + center_y
        return torch.cartesian_prod(xs, ys)

    return torch.cat([cells(BAR, 0.0), cells(STEM, -STEM_OFFSET)]).to(device, dtype)


def _build():
    mujoco = import_mujoco()
    spec = studio(scale=0.6, table_half=0.45)
    spec.option.timestep = 0.002
    world = spec.worldbody
    for x, y, hx, hy in [(0, ARENA + 0.01, ARENA + 0.02, 0.01), (0, -ARENA - 0.01, ARENA + 0.02, 0.01),
                         (ARENA + 0.01, 0, 0.01, ARENA), (-ARENA - 0.01, 0, 0.01, ARENA)]:  # fmt: skip
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, 0.02], pos=[x, y, 0.02], rgba=list(WALL),
                       contype=2, conaffinity=2)  # fmt: skip

    pusher = world.add_body(name="pusher", pos=[0, 0, HEIGHT])
    for axis, name in (([1, 0, 0], "pusher_x"), ([0, 1, 0], "pusher_y")):
        pusher.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis, damping=20)
    pusher.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[PUSHER, HEIGHT, 0], rgba=list(PUSHER_COLOR), mass=0.2,
                    contype=2, conaffinity=2, friction=[0.4, 0.005, 0.0001])  # fmt: skip

    block = world.add_body(name="block", pos=[0, 0, HEIGHT])
    for axis, name in (([1, 0, 0], "block_x"), ([0, 1, 0], "block_y")):  # table friction, as damping and dry friction
        block.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis, damping=0.6, frictionloss=0.15)
    block.add_joint(
        name="block_angle", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], damping=0.003, frictionloss=0.0008
    )
    t = [
        {"type": mujoco.mjtGeom.mjGEOM_BOX, "size": [*BAR, HEIGHT], "pos": [0, 0, 0]},
        {"type": mujoco.mjtGeom.mjGEOM_BOX, "size": [*STEM, HEIGHT], "pos": [0, -STEM_OFFSET, 0]},
    ]
    for g in t:
        block.add_geom(**g, rgba=list(T_COLOR), mass=0.05, contype=2, conaffinity=2, friction=[0.4, 0.005, 0.0001])
    add_ghost(spec, t)

    for joint in ("pusher_x", "pusher_y"):
        spec.add_actuator(target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT, gainprm=[400] + [0] * 9,
                          biasprm=[0, -400, -40] + [0] * 7, biastype=mujoco.mjtBias.mjBIAS_AFFINE)  # fmt: skip

    eye = [0, 0, 1.2]
    world.add_camera(name="top", pos=eye, quat=look_at(eye, [0, 0, 0], up=(0, 1, 0)), fovy=VIEW,
                     proj=mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC)  # fmt: skip
    world.add_camera(name="front", pos=[0, -0.75, 0.6], quat=look_at([0, -0.75, 0.6], [0, -0.02, 0]), fovy=40)
    return spec.compile()
