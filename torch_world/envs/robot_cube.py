"""Single-cube pick-and-place in MuJoCo with a Universal Robots UR10e or a SO-101 arm; requires ``mujoco``."""

import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from torch_world.envs.legacy import LegacyEnvironment
from torch_world.envs.mujoco import TABLE, MujocoEnv, add_ghost, import_mujoco, look_at, studio


@dataclass(frozen=True)
class Robot:
    """An arm from MuJoCo Menagerie and the task geometry around it, in meters and radians."""

    arm: str  # model path inside Menagerie
    gripper: str | None  # gripper model attached at the arm's ``attachment_site``, if any
    site: str  # site at the gripper's working point
    grasp: tuple[float, float, float] | None  # grasp point in the site frame, when it is not the site itself
    settle: int  # steps the gripper needs to close before lifting
    joints: list[str]  # arm joints driven by inverse kinematics
    home: list[float]  # joint positions that start inverse kinematics, gripper pointing down
    approach: int | None  # site axis kept pointing down (a 5-DoF arm), or None to keep the full orientation
    wrist: tuple  # wrist camera eye, target and up vector in the site frame
    grip: tuple[float, float]  # gripper actuator commands when open and when closed
    workspace: tuple[tuple[float, float], tuple[float, float]]  # x and y ranges of cube and goal positions
    cube: float  # cube half size
    hover: float  # gripper height while carrying
    max_step: float  # gripper motion per step at full action
    radius: float  # success radius around the goal: one cube width
    camera: tuple  # front camera eye and target


MENAGERIE = "0059d4335f8156206f63a35662313385f7ad6d74"
RAW = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{commit}/{path}"

ROBOTS = {
    "ur10e": Robot(
        arm="universal_robots_ur10e/ur10e.xml",
        gripper="robotiq_2f85/2f85.xml",
        site="gripper/pinch",
        grasp=None,
        settle=2,
        joints=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        home=[-3.1416, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
        approach=None,
        wrist=((0.0, -0.09, -0.12), (0.0, 0.0, 0.06), (1.0, 0.0, 0.0)),
        grip=(0.0, 255.0),
        workspace=((0.55, 0.95), (-0.25, 0.25)),
        cube=0.025,
        hover=0.2,
        max_step=0.05,
        radius=0.05,
        camera=((1.15, -0.65, 1.05), (0.6, 0.0, 0.0)),
    ),
    "so101": Robot(
        arm="robotstudio_so101/so101.xml",
        gripper=None,
        site="gripperframe",
        grasp=(-0.008, 0.0, -0.008),
        settle=5,
        joints=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        home=[0.0, 0.0, 0.0, 1.57, 0.0],
        approach=0,
        wrist=((-0.06, 0.0, 0.035), (0.04, 0.0, 0.0), (0.0, 0.0, 1.0)),
        grip=(1.5, -0.1),
        workspace=((0.16, 0.3), (-0.1, 0.1)),
        cube=0.012,
        hover=0.08,
        max_step=0.02,
        radius=0.024,
        camera=((0.62, -0.33, 0.36), (0.2, 0.0, 0.04)),
    ),
}
CUBE = (0.16, 0.47, 0.84, 1.0)


class RobotCube(MujocoEnv, LegacyEnvironment):
    """Single-cube pick-and-place with a real arm in MuJoCo, after OGBench ``cube-single``, in a light studio scene.

    ``robot`` is ``"ur10e"`` (Universal Robots UR10e with a Robotiq 2F-85 gripper) or ``"so101"`` (the open-source
    SO-101 arm), both from MuJoCo Menagerie and downloaded once into ``root``. Actions ``(dx, dy, dz, grip)`` in
    ``[-1, 1]^4`` set the gripper's target up to ``max_step`` meters from its current position, solved by damped
    least-squares inverse kinematics with the gripper pointing down, and open (``grip > 0``) or close the gripper. Each
    step lasts ``control_dt`` seconds of simulation. Targets are relative to the gripper rather than to the previous
    target, and the arm is gravity-compensated so its joint servos hold their targets: the next state depends on the
    state, not on a hidden command, and resetting to a recorded state replays its actions (up to the joint
    velocities and finger positions, which the state omits). The state is ``(gripper x, y, z, opening, cube x, y, z)``.

    ``task`` sets what ``tasks``, ``distance``, ``success`` and ``position`` measure: ``"pick"`` carries the cube to a
    goal spot, ``"reach"`` moves the open gripper to a goal pose with the cube left in place.

    The batch is simulated sequentially on the CPU; observations are ``(B, 3, size, size)`` renders of the
    ``"front"`` camera (``"top"`` is an orthographic view of the workspace, ``"wrist"`` rides on the gripper).

    Park et al., "OGBench: Benchmarking Offline Goal-Conditioned RL", ICLR 2025.
    Paper: https://arxiv.org/abs/2410.20092
    Code: https://github.com/seohongpark/ogbench
    Models: https://github.com/google-deepmind/mujoco_menagerie
    """

    env_id = "RobotCube-v0"
    action_dim = 4
    state_dim = 7

    def __init__(
        self,
        robot: str = "ur10e",
        batch_size: int = 1,
        size: int = 64,
        camera: str = "front",
        control_dt: float = 0.1,
        max_step: float | None = None,
        task: str = "pick",
        root: str | Path | None = None,
        device=None,
        *,
        quality="high",
        render_size=None,
        max_episode_steps=100,
        observation="rgb",
        presentation_camera="overview",
    ):
        if task not in ("pick", "reach"):
            raise ValueError(f"task must be 'pick' or 'reach', not {task!r}")
        self.robot, self.config, self.task = robot, ROBOTS[robot], task
        config = self.config
        super().__init__(
            _build(robot, Path(root or Path.home() / ".cache" / "torch_world" / "menagerie")),
            batch_size,
            size,
            camera,
            device,
        )
        self.max_step, self.radius, self.half = max_step or config.max_step, config.radius, config.cube
        (self.x_range, self.y_range), self.hover = config.workspace, config.hover
        self.top_size = max(self.x_range[1] - self.x_range[0], self.y_range[1] - self.y_range[0]) + 0.1
        self.substeps = max(1, round(control_dt / self.model.opt.timestep))
        self.scratch = self.mujoco.MjData(self.model)
        self.site = self.model.site("grasp").id
        joints = [self.model.joint(name).id for name in config.joints]
        self.qpos, self.dofs = self.model.jnt_qposadr[joints], self.model.jnt_dofadr[joints]
        self.limits = self.model.jnt_range[joints].T
        self.arm = np.arange(len(joints))
        self.cube_qpos = self.model.joint("cube").qposadr[0]
        self.targets = np.zeros((batch_size, 3))
        self.opening = np.ones(batch_size)
        self.closed_for = np.zeros(batch_size)
        self.down = None
        self.state = torch.zeros(batch_size, self.state_dim, device=device)
        self.env_id = "RobotCube-v0" if robot == "ur10e" else "RobotCubeSO101-v0"
        self._configure_benchmark(
            quality=quality,
            render_size=render_size,
            max_episode_steps=max_episode_steps,
            observation=observation,
            presentation_camera=presentation_camera,
            robot=robot,
            camera=camera,
            control_dt=control_dt,
            max_step=self.max_step,
            task=task,
            root=root,
        )

    # simulation

    def reset(self, state: Tensor | None = None, generator: torch.Generator | None = None) -> Tensor:
        """A random cube on the table and the open gripper hovering above the workspace, or the given ``state``."""
        if state is None:
            u = torch.rand(self.batch_size, 4, generator=generator).double().numpy()
            gripper = np.stack(
                [_span(self.x_range, u[:, 0]), _span(self.y_range, u[:, 1]), np.full(len(u), self.hover)], 1
            )
            cube = np.stack([_span(self.x_range, u[:, 2]), _span(self.y_range, u[:, 3]), np.full(len(u), self.half)], 1)
            opening = np.ones(len(u))
        else:
            s = state.double().cpu().numpy()
            gripper, opening, cube = s[:, :3], s[:, 3], s[:, 4:7]
        for b, d in enumerate(self.data):
            self._place(d, np.concatenate([gripper[b], [opening[b]], cube[b]]))
        self.targets, self.opening = gripper.copy(), opening.copy()
        self.closed_for = np.zeros(self.batch_size)
        self._sync()
        return self.render()

    def step(self, action: Tensor) -> Tensor:
        a = action.clamp(-1, 1).double().cpu().numpy()
        low = np.array([self.x_range[0] - 0.05, self.y_range[0] - 0.05, self.half * 0.6])
        high = np.array([self.x_range[1] + 0.05, self.y_range[1] + 0.05, self.hover + 0.1])
        gripper = np.stack([d.site_xpos[self.site] for d in self.data])
        self.targets = np.clip(gripper + self.max_step * a[:, :3], low, high)
        self.opening = np.clip(self.opening + a[:, 3], 0, 1)
        self.closed_for = np.where(self.opening < 0.5, self.closed_for + 1, 0)
        for b, d in enumerate(self.data):
            d.ctrl[self.arm] = self._ik(d.qpos[self.qpos], self.targets[b])
            d.ctrl[-1] = _span(self.config.grip, 1 - self.opening[b])
            self.mujoco.mj_step(self.model, d, self.substeps)
        self._sync()
        return self.render()

    def _place(self, d, state: np.ndarray):
        mujoco = self.mujoco
        gripper, opening, cube = state[:3], state[3], state[4:7]
        mujoco.mj_resetData(self.model, d)
        d.qpos[self.qpos] = self._ik(np.array(self.config.home), gripper, iterations=200)
        d.ctrl[self.arm] = d.qpos[self.qpos]
        d.ctrl[-1] = _span(self.config.grip, 1 - opening)
        d.qpos[self.cube_qpos : self.cube_qpos + 7] = [*cube, 1, 0, 0, 0]
        mujoco.mj_forward(self.model, d)

    def _ik(self, q: np.ndarray, target: np.ndarray, iterations: int = 20, damping: float = 1e-2) -> np.ndarray:
        """Joint positions putting the gripper site at ``target`` and pointing it down."""
        mujoco, m, d = self.mujoco, self.model, self.scratch
        if self.down is None:
            d.qpos[self.qpos] = self.config.home
            mujoco.mj_kinematics(m, d)
            self.down = d.site_xmat[self.site].reshape(3, 3).copy()
        q = q.copy()
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iterations):
            d.qpos[self.qpos] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            rot = d.site_xmat[self.site].reshape(3, 3)
            axes = range(3) if self.config.approach is None else [self.config.approach]
            turn = sum(np.cross(rot[:, i], self.down[:, i]) for i in axes)
            err = np.concatenate([target - d.site_xpos[self.site], 0.5 * turn])
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site)
            jac = np.concatenate([jacp, jacr])[:, self.dofs]
            q = np.clip(q + jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), err), *self.limits)
        return q

    def _state(self, b: int, d) -> list[float]:
        return [*d.site_xpos[self.site], self.opening[b], *d.qpos[self.cube_qpos : self.cube_qpos + 3]]

    def _ghost(self, goal: np.ndarray):
        return goal[4:7], [1, 0, 0, 0]

    # tasks and scoring

    def tasks(self, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
        """Random starts, and goals the scripted expert reaches from them: the cube carried to a random spot with the
        gripper backed off (``"pick"``, run until every cube is placed, up to 60 steps), or the open gripper moved to a
        random pose above the workspace (``"reach"``)."""
        self.reset(generator=generator)
        start = self.state.clone()
        if self.task == "pick":
            target = self.sample_targets(generator=generator)
            for _ in range(60):
                self.step(self.expert(target))
                done = _placed(
                    self.state, torch.cat([self.state[:, :4], target, self.state[:, 6:]], -1), self.radius, self.half
                )
                if (done & (self.state[:, 2] > self.hover - 0.02)).all():
                    break
            goal = self.state.clone()
        else:
            aim = self.reach_targets(generator)
            for _ in range(12):
                self.step(self.scripted(torch.cat([aim, self.state[:, 3:]], -1)))
            goal = self.state.clone()
        self.reset(start)
        return start, goal

    def reach_targets(self, generator: torch.Generator | None = None) -> Tensor:
        """Random gripper positions ``(B, 3)`` above the workspace, the aims of ``"reach"`` tasks."""
        u = torch.rand(self.batch_size, 3, generator=generator)
        return torch.stack(
            [_span(self.x_range, u[:, 0]), _span(self.y_range, u[:, 1]), _span((3 * self.half, self.hover), u[:, 2])],
            -1,
        ).to(self.state)

    def scripted(self, goal: Tensor) -> Tensor:
        """Clean expert actions ``(B, 4)`` toward goal states ``(B, 7)`` of the current ``task``."""
        if self.task == "pick":
            return self.expert(goal[:, 4:6])
        move = (goal[:, :3] - self.state[:, :3].to(goal)) / self.max_step
        return torch.cat([move.clamp(-1, 1), torch.ones_like(goal[:, :1])], -1)

    def position(self, states: Tensor) -> Tensor:
        """The cube's (``"pick"``) or the gripper's (``"reach"``) position on the table in the unit square of the
        ``"top"`` camera (x right, y down), which spans ``top_size`` meters."""
        x = 4 if self.task == "pick" else 0
        cx, cy = sum(self.x_range) / 2, sum(self.y_range) / 2
        return 0.5 - torch.stack([states[..., x + 1] - cy, states[..., x] - cx], dim=-1) / self.top_size

    def object_position(self, states: Tensor) -> Tensor:
        """Position in meters of what the task moves to its goal, for path-based metrics such as SPL: the cube on the
        table plane (``"pick"``; lifting it is not a detour) or the gripper in 3D (``"reach"``)."""
        return states[..., 4:6] if self.task == "pick" else states[..., :3]

    def distance(self, states: Tensor, goal: Tensor) -> Tensor:
        """Distance in meters between the cube and its goal spot on the table (``"pick"``), or between the gripper and
        its goal position (``"reach"``), for states ``(B, ..., 7)`` and goals ``(B, 7)``."""
        goal = goal.view(len(goal), *[1] * (states.dim() - 2), -1)
        if self.task == "reach":
            return (states[..., :3] - goal[..., :3]).norm(dim=-1)
        return (states[..., 4:6] - goal[..., 4:6]).norm(dim=-1)

    def success(self, states: Tensor, goal: Tensor) -> Tensor:
        """The cube rests on the table within ``radius`` of its goal (``"pick"``), or the gripper is within ``radius``
        of its goal (``"reach"``)."""
        if self.task == "reach":
            return self.distance(states, goal) <= self.radius
        return _placed(states, goal, self.radius, self.half)

    def sample_targets(self, distance: tuple[float, float] | None = None, generator=None) -> Tensor:
        """Goal cube positions ``(B, 2)`` inside the workspace, at least ``distance[0]`` and at most ``distance[1]`` away."""
        lo, hi = distance or (2 * self.half + 0.02, 0.5 * (self.x_range[1] - self.x_range[0]) + 0.1)
        cube = self.state[:, 4:6].cpu()
        targets = cube.clone()
        todo = torch.ones(self.batch_size, dtype=torch.bool)
        while todo.any():
            u = torch.rand(self.batch_size, 2, generator=generator)
            angle, radius = u[:, 0] * 2 * torch.pi, lo + (hi - lo) * u[:, 1]
            candidate = cube + radius[:, None] * torch.stack([angle.cos(), angle.sin()], -1)
            inside = (candidate[:, 0] >= self.x_range[0]) & (candidate[:, 0] <= self.x_range[1])
            inside &= (candidate[:, 1] >= self.y_range[0]) & (candidate[:, 1] <= self.y_range[1])
            targets[inside & todo], todo = candidate[inside & todo], todo & ~inside
        return targets.to(self.state.device)

    def expert(self, target: Tensor, state: Tensor | None = None) -> Tensor:
        """Scripted pick-and-place actions ``(B, 4)`` that carry the cube to ``target`` ``(B, 2)`` and back off.

        Motion is planned from the gripper position of ``state``, by default the running episode's.
        """
        s = self.state if state is None else state
        ee, opening, cube = s[:, :3], s[:, 3], s[:, 4:7]
        settled = torch.as_tensor(self.closed_for >= self.config.settle, device=s.device) if state is None else True
        grasped = (opening < 0.5) & ((ee - cube).norm(dim=-1) < 2 * self.half) & settled
        placed = ~grasped & ((cube[:, :2] - target).norm(dim=-1) < self.radius) & (cube[:, 2] < self.half + 0.005)
        goal_xy = torch.where((grasped | placed)[:, None], target, cube[:, :2])
        above = (ee[:, :2] - goal_xy).norm(dim=-1) < 0.15 * self.radius + 0.002
        low = (ee[:, 2] - self.half).abs() < 0.005
        down = above & ~placed
        z = torch.where(down, torch.full_like(ee[:, 2], self.half), torch.full_like(ee[:, 2], self.hover))
        xy = torch.where(placed[:, None], ee[:, :2], goal_xy)
        move = ((torch.cat([xy, z[:, None]], -1) - ee) / self.max_step).clamp(-1, 1)
        close = torch.where(grasped, ~(down & low), down & low & ~grasped)
        return torch.cat([move, torch.where(close, -1.0, 1.0)[:, None]], dim=-1)

    def solve(self, target: Tensor, steps: int) -> Tensor:
        """Run the clean expert for ``steps``; returns the final states, used as goals."""
        for _ in range(steps):
            self.step(self.expert(target))
        return self.state.clone()

    def collect(
        self,
        num_steps: int,
        noise: float = 0.04,
        cameras: list[str] | None = None,
        generator: torch.Generator | None = None,
    ) -> dict:
        """Play data from the noisy expert, retargeted after every placement, as in OGBench's ``play`` datasets.

        Returns ``pixels`` ``(B, T, 3, size, size)`` (or ``(B, T, V, 3, size, size)`` for a list of ``cameras``),
        ``actions`` ``(B, T, 4)`` and ``states`` ``(B, T, 7)``; ``actions[:, t]`` is applied at ``states[:, t]``.
        ``noise`` is the standard deviation added to the expert's actions.
        """

        def observe():
            if cameras is None:
                return self.render()
            return torch.stack([self.render(camera=c) for c in cameras], dim=1)

        self.reset(generator=generator)
        pixels, states, actions = [observe()], [self.state], []
        target = self.sample_targets(generator=generator)
        for _ in range(num_steps - 1):
            n = torch.randn(self.batch_size, 4, generator=generator).to(self.state)
            action = (self.expert(target) + noise * n).clamp(-1, 1)
            actions.append(action)
            self.step(action)
            pixels.append(observe())
            states.append(self.state)
            done = _placed(
                self.state, torch.cat([self.state[:, :4], target, self.state[:, 6:]], -1), self.radius, self.half
            )
            target = torch.where(done[:, None], self.sample_targets(generator=generator), target)
        actions.append(torch.zeros_like(actions[-1]))
        return {"pixels": torch.stack(pixels, 1), "actions": torch.stack(actions, 1), "states": torch.stack(states, 1)}


def _placed(states: Tensor, goal: Tensor, radius: float, half: float) -> Tensor:
    goal = goal.view(len(goal), *[1] * (states.dim() - 2), -1)
    return ((states[..., 4:6] - goal[..., 4:6]).norm(dim=-1) <= radius) & (states[..., 6] < half + 0.01)


def _span(bounds, u):
    return bounds[0] + (bounds[1] - bounds[0]) * u


def fetch(path: str, root: Path) -> Path:
    """Download a Menagerie model file and every mesh and texture it references, once."""
    file = root / MENAGERIE[:8] / path
    if not file.exists():
        _download(path, file)
    tree = ET.parse(file)
    compiler = tree.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    texturedir = compiler.get("texturedir", meshdir) if compiler is not None else ""
    for e in tree.iter():
        if e.tag in ("mesh", "texture") and e.get("file"):
            rel = str(Path(path).parent / (meshdir if e.tag == "mesh" else texturedir) / e.get("file"))
            if not (root / MENAGERIE[:8] / rel).exists():
                _download(rel, root / MENAGERIE[:8] / rel)
    return file


def _download(path: str, file: Path):
    file.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(RAW.format(commit=MENAGERIE, path=path), file)


def _build(robot: str, root: Path, scene=None):
    mujoco = import_mujoco()
    config = ROBOTS[robot]
    (x0, x1), (y0, y1) = config.workspace
    reach = x1 + 0.25
    spec = studio(scale=reach, table_center=(reach * 0.4, 0), table_half=reach * 1.2)
    world = spec.worldbody
    world.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, 0.365], pos=[reach * 0.5, 0, -0.405], rgba=[*TABLE, 1]
    )
    if scene is None:
        cube = world.add_body(name="cube", pos=[(x0 + x1) / 2, 0, config.cube])
        cube.add_freejoint(name="cube")
        h = config.cube
        cube.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[h, h, h],
            rgba=list(CUBE),
            mass=0.05 if robot == "ur10e" else 0.01,
            friction=[1.5, 0.01, 0.001],
        )
        add_ghost(spec, [{"type": mujoco.mjtGeom.mjGEOM_BOX, "size": [h, h, h]}])
    else:
        scene(spec)
    eye, target = config.camera
    world.add_camera(name="front", pos=list(eye), quat=look_at(eye, target), fovy=40)
    side = max(x1 - x0, y1 - y0) + 0.1
    world.add_camera(
        name="top",
        pos=[(x0 + x1) / 2, (y0 + y1) / 2, 1.5],
        quat=look_at([(x0 + x1) / 2, (y0 + y1) / 2, 1.5], [(x0 + x1) / 2 - 1e-4, (y0 + y1) / 2, 0]),
        fovy=side,
        proj=mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC,
    )

    arm = mujoco.MjSpec.from_file(str(fetch(config.arm, root)))
    parts = [spec, arm]
    if config.gripper is not None:
        parts.append(mujoco.MjSpec.from_file(str(fetch(config.gripper, root))))
    for part in parts[1:]:  # the scene's lights only
        for light in list(part.lights):
            part.delete(light)
    for part in parts:  # one set of solver options, suited to grasping, so attaching causes no conflicts
        part.option.timestep, part.option.impratio, part.option.iterations, part.option.ls_iterations = (
            0.002,
            10,
            100,
            50,
        )
        part.option.cone, part.option.integrator = (
            mujoco.mjtCone.mjCONE_ELLIPTIC,
            mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        )
    if config.gripper is not None:
        arm.attach(parts[2], prefix="gripper/", site=arm.site("attachment_site"))
    for link in arm.bodies:  # gravity compensation of arm and gripper, so the joint servos hold their targets
        link.gravcomp = 1
    resolved = arm.compile()
    site_pos, site_quat = resolved.site(config.site).pos, resolved.site(config.site).quat
    body = arm.site(config.site).parent

    def in_site_frame(pos, quat=(1.0, 0.0, 0.0, 0.0)):
        p, q = np.zeros(3), np.zeros(4)
        mujoco.mju_rotVecQuat(p, np.array(pos, dtype=float), site_quat)
        mujoco.mju_mulQuat(q, site_quat, np.array(quat, dtype=float))
        return (site_pos + p).tolist(), q.tolist()

    pos, quat = in_site_frame(config.grasp or (0.0, 0.0, 0.0))
    body.add_site(name="grasp", pos=pos, quat=quat)
    eye, target, up = config.wrist
    pos, quat = in_site_frame(eye, look_at(eye, target, up))
    body.add_camera(name="wrist", pos=pos, quat=quat, fovy=75)
    spec.attach(arm, prefix="", frame=world.add_frame(pos=[0, 0, 0]))
    return spec.compile()
