"""SO101 chess episodes with wrist observations and privileged simulation metrics."""

from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch_world.envs.mujoco import MujocoEnv, look_at
from torch_world.envs.robot_cube import ROBOTS, _build
from torch_world.sim2real.config import ChessTaskConfig, RandomizationConfig, SO101Config
from torch_world.sim2real.control import Observation, SO101Controller
from torch_world.sim2real.scene import chess_scene, square_position


class SO101Chess:
    """One physical-contact chess piece on an empty board, batch size one.

    RGB/proprioception are deployable observations. ``info`` and ``expert`` use
    simulator-only piece pose; neither is exposed to a learned policy.
    """

    action_dim = 4

    def __init__(
        self, config=None, task=None, *, piece="pawn", color="white", randomization=None, calibration=None, root=None
    ):
        self.config, self.task = config or SO101Config(), task or ChessTaskConfig()
        if piece not in self.task.pieces or color not in ("white", "black"):
            raise ValueError("Unknown piece or color")
        self.piece, self.color = piece, color
        self.randomization = randomization or RandomizationConfig(enabled=False)
        self.calibration = calibration
        self.model = _build(
            "so101", Path(root or Path.home() / ".cache/torch_world/menagerie"), chess_scene(self.task, piece, color)
        )
        # Recolor printed shells, preserving the dark servo housings and metal parts.
        for index in range(self.model.nmat):
            name = self.model.material(index).name
            if "so101" in name and not name.startswith("sts3215"):
                self.model.mat_rgba[index] = [*self.config.arm_color, 1.0]
                self.model.mat_specular[index] = 0.15
                self.model.mat_shininess[index] = 0.3
        self.model.vis.quality.offsamples = 8
        self.world = MujocoEnv(self.model, 1, self.config.size, "wrist")
        self.mj, self.data = self.world.mujoco, self.world.data[0]
        view = self.config.wrist_view
        site = self.model.site("gripperframe")
        offset, quat = np.zeros(3), np.zeros(4)
        self.mj.mju_rotVecQuat(offset, np.asarray(view.eye), site.quat)
        self.mj.mju_mulQuat(quat, site.quat, np.asarray(look_at(view.eye, view.target, view.up)))
        self.model.camera("wrist").pos = site.pos + offset
        self.model.camera("wrist").quat = quat
        self.model.camera("wrist").fovy = view.fovy
        self.controller = SO101Controller(self.config, self.model)
        self.piece_body = self.model.body("piece").id
        self.piece_qpos = self.model.joint("piece").qposadr[0]
        self.grip_joint = self.model.joint("gripper").qposadr[0]
        self.grip_actuator = self.model.actuator("gripper").id
        self.geoms = np.flatnonzero(self.model.geom_bodyid == self.piece_body)
        if calibration is not None:
            calibration.apply_camera(self.model, "wrist")
        self._nominal = {
            name: getattr(self.model, name).copy()
            for name in (
                "geom_size",
                "geom_pos",
                "geom_rgba",
                "geom_friction",
                "body_mass",
                "body_inertia",
                "actuator_gainprm",
                "actuator_biasprm",
                "light_diffuse",
                "cam_pos",
                "cam_quat",
            )
        }
        self.substeps = max(1, round(self.config.control_dt / self.model.opt.timestep))
        if abs(self.substeps * self.model.opt.timestep - self.config.control_dt) > 1e-7:
            raise ValueError("control_dt must be a multiple of the simulation timestep")
        self.rng = np.random.default_rng(self.config.seed)
        self.steps, self.stable, self.phase = 0, 0, 0
        self.done = False
        self.episode_parameters = {}
        self.last_command = None
        self._reachable = None

    @property
    def joints(self):
        return self.data.qpos[self.controller.indices].copy()

    @property
    def opening(self):
        closed, opened = ROBOTS["so101"].grip[1], ROBOTS["so101"].grip[0]
        return float(np.clip((self.data.qpos[self.grip_joint] - closed) / (opened - closed), 0, 1))

    @property
    def position(self):
        return self.data.xpos[self.piece_body].copy()

    @property
    def tool_position(self):
        return self.data.site_xpos[self.controller.site].copy()

    @property
    def grasp_height(self):
        return self.task.pieces[self.piece].grasp_height * self.episode_parameters.get("dimension_scale", 1)

    @property
    def hover(self):
        return min(self.config.workspace_high[2] - 0.01, self.task.board_origin[2] + self.task.hover_height)

    def reachable_squares(self):
        """Check approach and grasp IK, including piece-to-square clearance."""
        if self._reachable is None:
            found = []
            for file in "abcdefgh":
                for rank in "12345678":
                    square = file + rank
                    base = square_position(self.task, square)
                    targets = [np.r_[base[:2], z] for z in np.linspace(self.hover, base[2] + self.grasp_height, 12)]
                    q = np.array(ROBOTS["so101"].home)
                    valid = True
                    for target in targets:
                        if np.any(target < self.config.workspace_low) or np.any(target > self.config.workspace_high):
                            valid = False
                            break
                        q, error = self.controller.solve(q, target, iterations=160)
                        if error > min(0.002, self.config.ik_tolerance / 2):
                            valid = False
                            break
                        try:
                            self.controller.check_clearance(q, q, 1.0)
                        except ValueError:
                            valid = False
                            break
                    if valid:
                        found.append(square)
            self._reachable = found
        return list(self._reachable)

    def _transfer_reachable(self, source, target):
        start, end = square_position(self.task, source), square_position(self.task, target)
        q = np.array(ROBOTS["so101"].home)
        for fraction in np.linspace(0, 1, 12):
            position = np.r_[(start + fraction * (end - start))[:2], self.hover]
            previous = q
            q, error = self.controller.solve(q, position, iterations=160)
            if error > min(0.002, self.config.ik_tolerance / 2):
                return False
            try:
                self.controller.check_clearance(previous, q, 0.0)
            except ValueError:
                return False
        return True

    def _randomize(self):
        for name, value in self._nominal.items():
            getattr(self.model, name)[:] = value
        r = self.randomization
        parameters = {
            key: float(self.rng.uniform(*getattr(r, key))) if r.enabled else 1.0
            for key in ("dimension_scale", "mass_scale", "friction_scale", "gain_scale", "light_scale")
        }
        scale = parameters["dimension_scale"]
        if self.task.pieces[self.piece].diameter * scale >= self.task.square_size:
            raise ValueError("Randomized piece no longer fits inside its square")
        self.model.geom_size[self.geoms] *= scale
        self.model.geom_pos[self.geoms] *= scale
        self.model.body_mass[self.piece_body] *= parameters["mass_scale"]
        self.model.body_inertia[self.piece_body] *= parameters["mass_scale"] * scale**2
        self.model.geom_friction[self.geoms, 0] *= parameters["friction_scale"]
        self.model.actuator_gainprm[:, 0] *= parameters["gain_scale"]
        self.model.actuator_biasprm[:, 1:3] *= parameters["gain_scale"]
        self.model.light_diffuse[:] *= parameters["light_scale"]
        camera = self.model.camera("wrist").id
        translation = self.rng.uniform(-r.camera_translation, r.camera_translation, 3) if r.enabled else np.zeros(3)
        rotation = self.rng.uniform(-r.camera_rotation, r.camera_rotation, 3) if r.enabled else np.zeros(3)
        self.model.cam_pos[camera] += translation
        angle = np.linalg.norm(rotation)
        delta, quat = np.zeros(4), np.zeros(4)
        self.mj.mju_axisAngle2Quat(delta, rotation / angle if angle else np.array([1.0, 0, 0]), angle)
        self.mj.mju_mulQuat(quat, self.model.cam_quat[camera].copy(), delta)
        self.model.cam_quat[camera] = quat
        appearance = float(self.rng.uniform(0.8, 1.15)) if r.enabled else 1.0
        self.model.geom_rgba[self.geoms, :3] = np.clip(self.model.geom_rgba[self.geoms, :3] * appearance, 0, 1)
        parameters.update(
            camera_translation=translation.tolist(),
            camera_rotation=rotation.tolist(),
            appearance=appearance,
            action_delay=int(self.rng.integers(r.max_action_delay + 1)) if r.enabled else 0,
            observation_delay=int(self.rng.integers(r.max_observation_delay + 1)) if r.enabled else 0,
        )
        self.mj.mj_setConst(self.model, self.data)
        self.episode_parameters = parameters
        self._reachable = None

    def reset(self, *, seed=None, source=None, target=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._randomize()
        squares = self.reachable_squares()
        if len(squares) < 2:
            raise ValueError("Fewer than two reachable squares; adjust board pose/dimensions")
        if (source is not None and source not in squares) or (target is not None and target not in squares):
            raise ValueError(f"Source/target must be distinct reachable squares: {squares}")
        pairs = [
            (a, b)
            for a in squares
            for b in squares
            if a != b and (source is None or a == source) and (target is None or b == target)
        ]
        for index in self.rng.permutation(len(pairs)):
            a, b = pairs[index]
            if self._transfer_reachable(a, b):
                source, target = a, b
                break
        else:
            raise ValueError(f"Source/target must be distinct reachable squares with a clear transfer: {squares}")
        self.source, self.target_square = source, target
        self.target_position = square_position(self.task, target)
        self.mj.mj_resetData(self.model, self.data)
        base = square_position(self.task, source)
        initial = np.r_[base[:2], self.hover]
        q, residual = self.controller.solve(ROBOTS["so101"].home, initial, iterations=200)
        if residual > self.config.ik_tolerance:
            raise ValueError("Cannot initialize approach pose")
        self.reference_joints = q.copy()
        self.data.qpos[self.controller.indices] = q
        self.data.ctrl[self.controller.actuators] = q
        self.data.qpos[self.grip_joint] = self.data.ctrl[self.grip_actuator] = ROBOTS["so101"].grip[0]
        self.data.qpos[self.piece_qpos : self.piece_qpos + 7] = [*base, 1, 0, 0, 0]
        self.mj.mj_forward(self.model, self.data)
        self.controller.reset(self.joints, self.opening)
        self.steps, self.stable, self.phase, self.phase_steps = 0, 0, 0, 0
        self._grip_stable, self._previous_opening = 0, self.opening
        self.done, self.lifted, self.last_command = False, False, None
        self.carry_offset = np.array([0.0, 0.0, -self.grasp_height])
        self._actions = deque([np.zeros(4) for _ in range(self.episode_parameters["action_delay"])])
        first = self.observe()
        self._observations = deque([first for _ in range(self.episode_parameters["observation_delay"])])
        return first

    def render_view(self, *, size=None, native=False):
        """Render the wrist's native aspect; resize identically for square model inputs."""
        width = size or self.config.size
        aspect = (
            self.calibration.resolution[0] / self.calibration.resolution[1]
            if self.calibration is not None
            else self.config.wrist_view.aspect
        )
        height = max(1, round(width / aspect))
        key = (width, height, "wrist")
        if key not in self.world.renderers:
            self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, width)
            self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, height)
            self.world.renderers[key] = self.mj.Renderer(self.model, height, width)
        renderer = self.world.renderers[key]
        renderer.update_scene(self.data, "wrist", scene_option=self.world.hidden)
        rgb = torch.from_numpy(renderer.render().copy()).permute(2, 0, 1).float() / 255
        if not native and height != width:
            rgb = F.interpolate(rgb[None], (width, width), mode="bilinear", antialias=True)[0]
        return rgb

    def observe(self, *, size=None, native=False):
        """Read pixels at model resolution, or explicitly render a larger presentation view."""
        rgb = self.render_view(size=size, native=native)
        if self.randomization.enabled and self.randomization.pixel_noise:
            rgb = (rgb + torch.from_numpy(self.rng.normal(0, self.randomization.pixel_noise, rgb.shape)).float()).clamp(
                0, 1
            )
        target = self.tool_position if self.controller.target is None else self.controller.target
        commanded = torch.tensor(np.r_[target, self.controller.opening], dtype=torch.float32)
        return Observation(
            rgb,
            torch.tensor(self.joints, dtype=torch.float32),
            self.opening,
            float(self.data.time),
            commanded=commanded,
        )

    def goal(self, *, size=None, native=False):
        """Goal at the initial reference wrist pose, with piece upright at destination."""
        saved_qpos, saved_qvel, saved_ctrl = self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy()
        try:
            self.data.qpos[self.controller.indices] = self.reference_joints
            self.data.qpos[self.grip_joint] = ROBOTS["so101"].grip[0]
            self.data.qpos[self.piece_qpos : self.piece_qpos + 7] = [*self.target_position, 1, 0, 0, 0]
            self.data.qvel[:] = 0
            self.mj.mj_forward(self.model, self.data)
            goal = self.observe(size=size, native=native)
            goal.commanded = torch.tensor(np.r_[self.tool_position, self.opening], dtype=torch.float32)
            return goal
        finally:
            self.data.qpos[:], self.data.qvel[:], self.data.ctrl[:] = saved_qpos, saved_qvel, saved_ctrl
            self.mj.mj_forward(self.model, self.data)

    def info(self):
        rotation = self.data.xmat[self.piece_body].reshape(3, 3)
        distance = float(np.linalg.norm(self.position[:2] - self.target_position[:2]))
        margin = (
            self.task.square_size / 2
            - self.task.pieces[self.piece].diameter * self.episode_parameters["dimension_scale"] / 2
        )
        placed = (
            distance <= min(self.task.placement_tolerance, margin)
            and abs(self.position[2] - self.target_position[2]) < 0.003
            and rotation[2, 2] >= np.cos(self.task.upright_tolerance)
            and self.opening > 0.8
            and self.tool_position[2] > self.hover - 0.015
            and np.linalg.norm(self.joints - self.reference_joints) < 0.12
        )
        return {
            "success": self.stable >= self.task.settle_steps,
            "placed": bool(placed),
            "distance": distance,
            "upright": bool(rotation[2, 2] >= np.cos(self.task.upright_tolerance)),
            "lifted": bool(self.lifted),
            "truncated": self.steps >= self.task.max_steps,
            "piece_position": self.position.tolist(),
        }

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode ended; call reset before stepping")
        a = np.asarray(torch.as_tensor(action).detach().cpu(), dtype=float)
        if a.shape != (4,) or not np.isfinite(a).all():
            raise ValueError("Expected finite action (4,)")
        self._actions.append(a)
        action = self._actions.popleft()
        command = self.controller.command(action, self.joints, self.opening)
        self.data.ctrl[self.controller.actuators] = command.joints
        self.data.ctrl[self.grip_actuator] = -0.1 + 1.6 * command.opening
        self.mj.mj_step(self.model, self.data, self.substeps)
        self.last_command = command
        self.steps += 1
        self.lifted |= self.position[2] > self.task.board_origin[2] + 0.015
        self.stable = self.stable + 1 if self.info()["placed"] and self.lifted else 0
        self.done = self.stable >= self.task.settle_steps or self.steps >= self.task.max_steps
        self._observations.append(self.observe())
        return self._observations.popleft()

    def expert(self, *, transition="timed"):
        """Simulation-only phase controller; never used by a learned hardware policy."""
        if transition not in ("timed", "observed"):
            raise ValueError("Expert transition must be timed or observed")
        source, dest = square_position(self.task, self.source), self.target_position
        height = self.grasp_height
        reference = self.controller.pose(self.reference_joints)
        if self.phase in (3, 4):
            self.carry_offset = self.position - self.tool_position
        place = dest - self.carry_offset
        # Withdraw along the gripper's opening axis before lifting past the
        # piece head. The carried piece offset also contains lateral slip and
        # is not a reliable jaw-clearance direction.
        away = -self.data.site_xmat[self.controller.site].reshape(3, 3)[:2, 2]
        away = away / max(np.linalg.norm(away), 1e-6)
        clearance = self.task.pieces[self.piece].diameter * self.episode_parameters["dimension_scale"] * 0.65
        retreat = place + np.r_[away * clearance, 0.0]
        targets = [
            source + [0, 0, height],
            source + [0, 0, height],
            np.r_[source[:2], self.hover],
            np.r_[place[:2], self.hover],
            place,
            place,
            retreat,
            np.r_[retreat[:2], self.hover],
            reference,
        ]
        close = self.phase in (1, 2, 3, 4)
        target = targets[self.phase]
        self.phase_steps += 1
        settled = abs(self.opening - self._previous_opening) < 0.003
        endpoint = self.opening < 0.05 if close else self.opening > 0.98
        self._grip_stable = self._grip_stable + 1 if settled and endpoint else 0
        self._previous_opening = self.opening
        reached = np.linalg.norm(self.tool_position - target) < (0.001 if self.phase == 4 else 0.004)
        ready = self.phase_steps >= 12 if transition == "timed" else self._grip_stable >= 3
        if reached and self.phase < 8 and (self.phase not in (1, 5) or ready):
            if self.phase == 2:
                self.carry_offset = self.position - self.tool_position
            self.phase += 1
            self.phase_steps = 0
            self._grip_stable = 0
        move = np.clip((target - self.controller.target) / self.config.max_step, -1, 1)
        return torch.tensor(np.r_[move, -0.15 if close else 0.15], dtype=torch.float32)

    def settings(self):
        return {
            "controller": asdict(self.config),
            "task": asdict(self.task),
            "randomization": asdict(self.randomization),
            "piece": self.piece,
            "color": self.color,
        }

    def close(self):
        for renderer in self.world.renderers.values():
            renderer.close()
        self.world.renderers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
