"""Batched execution of our original MuJoCo scenes."""

import torch

from torch_world.envs.benchmark import BenchmarkEnv, Transition
from torch_world.envs.physics_scenes import SCENES


class MujocoTask(BenchmarkEnv):
    """Original rigid-body tasks with normalized controls and restorable qpos/qvel.

    ``observation='state'`` avoids rendering cost. RGB uses one lazily allocated
    renderer per resolution, shared across batch lanes; rewards are vectorized.
    Free-joint quaternions are normalized by MuJoCo on reset.
    """

    def __init__(
        self,
        task=None,
        *,
        scene=None,
        batch_size=1,
        size=None,
        quality="high",
        render_size=None,
        device=None,
        max_episode_steps=300,
        observation="rgb",
        substeps=10,
        camera="task",
        goal_marker=False,
    ):
        super().__init__(batch_size, size, device, max_episode_steps, quality=quality, render_size=render_size)
        if observation not in ("rgb", "state") or substeps < 1:
            raise ValueError("observation must be rgb or state; substeps must be positive")
        try:
            import mujoco
            import numpy as np
        except ImportError as exc:
            raise ImportError("Install physics with pip install 'torch-world[lewam]'") from exc
        self.mj, self._np = mujoco, np
        if (task is None) == (scene is None):
            raise ValueError("Provide exactly one of task or scene")
        if goal_marker and task != "ObstacleArm":
            raise ValueError("goal_marker is currently supported for ObstacleArm only")
        self.goal_marker = goal_marker
        self.visual_goal = None
        self.scene = SCENES[task]() if scene is None else scene
        self.model = mujoco.MjModel.from_xml_string(self.scene.xml(), assets=self.scene.assets())
        self.data = [mujoco.MjData(self.model) for _ in range(batch_size)]
        self._scratch = mujoco.MjData(self.model)
        self._target_scratch = mujoco.MjData(self.model) if goal_marker else None
        self._renderers = {}
        self.observation, self.substeps = observation, substeps
        self.action_dim, self.state_dim = self.model.nu, self.model.nq + self.model.nv
        self._indices = torch.tensor(self.scene.indices, device=self.device)
        # qpos and qvel addresses differ after free/ball joints (quaternions).
        velocities = set()
        for index in self.scene.indices:
            joint = int(self._np.searchsorted(self.model.jnt_qposadr, index, side="right") - 1)
            offset = index - self.model.jnt_qposadr[joint]
            dof = int(self.model.jnt_dofadr[joint])
            kind = self.model.jnt_type[joint]
            if kind == mujoco.mjtJoint.mjJNT_BALL:
                velocities.update(range(dof, dof + 3))
            elif kind == mujoco.mjtJoint.mjJNT_FREE and offset >= 3:
                velocities.update(range(dof + 3, dof + 6))
            else:
                velocities.add(dof + int(offset))
        self._velocity_indices = torch.tensor(sorted(velocities), device=self.device)
        self._tolerance = torch.tensor(self.scene.tolerance, device=self.device)
        self._angular = torch.tensor([i in self.scene.angular for i in self.scene.indices], device=self.device)
        bounds = torch.tensor(self.scene.success_bounds, device=self.device).reshape(-1, 3)
        self._bound_indices = bounds[:, 0].long()
        self._bound_lower, self._bound_upper = bounds[:, 1], bounds[:, 2]
        self._mid = self.model.actuator_ctrlrange.mean(-1)
        self._scale = self._np.diff(self.model.actuator_ctrlrange, axis=-1).ravel() / 2
        self.goal = torch.zeros(batch_size, self.state_dim, device=self.device)
        self.goal[:, : self.model.nq] = torch.tensor(self.scene.target, device=self.device)
        self.state = self.goal.clone()
        if camera not in ("task", "overview"):
            raise ValueError("camera must be task or overview")
        self.camera = camera
        self.model.vis.global_.offwidth = max(self.size, self.render_size)
        self.model.vis.global_.offheight = max(self.size, self.render_size)
        self.model.vis.quality.offsamples = self.render_quality.samples
        # Frame the union of start and GT geometry with a fixed, reproducible camera.
        bounds = []
        for qpos in (self.scene.initial, self.scene.target):
            self._scratch.qpos[:] = qpos
            self.mj.mj_forward(self.model, self._scratch)
            lower, upper = self._geometry_bounds()
            bounds.extend((lower[None], upper[None]))
        points = self._np.concatenate(bounds)
        lower, upper = points.min(0), points.max(0)
        self._scene_bounds = lower, upper
        self._camera = self.mj.MjvCamera()
        self._camera.azimuth, self._camera.elevation = 90, -25
        tabletop = {"BallPlate", "PuckPush", "PuckSlide", "BoxGate", "PegInsertion", "PickPlace", "BlockStack"}
        if task in tabletop:
            self._camera.azimuth, self._camera.elevation = 115, -55
        if task in {"PickPlace", "BlockStack"}:
            self._camera.azimuth, self._camera.elevation = 135, -30
        if camera == "overview":
            self._camera.azimuth, self._camera.elevation = 135, -40
        self._fit_camera(self._camera, lower, upper)
        self._presentation_camera = self.mj.MjvCamera()
        self._presentation_camera.azimuth = self._camera.azimuth
        self._presentation_camera.elevation = self._camera.elevation
        self.model.vis.headlight.ambient[:] = 0.18
        self.model.vis.headlight.diffuse[:] = 0.5

    def set_visual_goal(self, goal=None):
        """Fix the visible collection target independently of the scored subgoal.

        Use the same target for observation and goal-image rendering, as in
        upstream LeWAM's Reacher replay. None follows the current task goal.
        The marker is render-only: no geometry, mass or contacts enter physics.
        """
        if goal is None:
            self.visual_goal = None
            return
        goal = torch.as_tensor(goal, dtype=torch.float32, device=self.device)
        if goal.shape != self.goal.shape or not torch.isfinite(goal).all():
            raise ValueError("visual goal must be finite and match the environment batch/state shape")
        self.visual_goal = goal.clone()

    def _geometry_bounds(self):
        positions = self._scratch.geom_xpos[1:]
        rotations = self._scratch.geom_xmat[1:].reshape(-1, 3, 3)
        centers = positions + self._np.einsum("nij,nj->ni", rotations, self.model.geom_aabb[1:, :3])
        extents = self._np.einsum("nij,nj->ni", abs(rotations), self.model.geom_aabb[1:, 3:])
        return (centers - extents).min(0), (centers + extents).max(0)

    def _fit_camera(self, camera, lower, upper):
        camera.lookat[:] = (lower + upper) / 2
        # Fit projected bounds to the actual field of view, including depth.
        azimuth, elevation = self._np.deg2rad([camera.azimuth, camera.elevation])
        direction = self._np.array(
            [
                self._np.cos(elevation) * self._np.cos(azimuth),
                self._np.cos(elevation) * self._np.sin(azimuth),
                -self._np.sin(elevation),
            ]
        )
        right = self._np.array([-self._np.sin(azimuth), self._np.cos(azimuth), 0.0])
        up = self._np.cross(direction, right)
        corners = self._np.array(
            [[x, y, z] for x in (lower[0], upper[0]) for y in (lower[1], upper[1]) for z in (lower[2], upper[2])]
        )
        centered = corners - camera.lookat
        tangent = self._np.tan(self._np.deg2rad(self.model.vis.global_.fovy) / 2)
        extent = self._np.maximum(abs(centered @ right), abs(centered @ up)) / tangent
        camera.distance = max(0.5, float((extent + centered @ direction).max()) * 1.15)
        if self.camera == "overview":
            camera.distance *= 1.2

    @property
    def render_settings(self):
        return {
            **super().render_settings,
            "follow_camera": self.scene.follow_camera,
            "directional_light": self.scene.directional_light,
        }

    def _restore(self, data, state):
        self.mj.mj_resetData(self.model, data)
        data.qpos[:] = state[: self.model.nq]
        data.qvel[:] = state[self.model.nq :]
        self.mj.mj_normalizeQuat(self.model, data.qpos)
        self.mj.mj_forward(self.model, data)

    def reset(self, state=None, generator=None, *, seed=None):
        if seed is not None and generator is not None:
            raise ValueError("Pass seed or generator, not both")
        if state is None:
            generator = torch.Generator().manual_seed(seed) if seed is not None else generator
            state = torch.zeros(self.batch_size, self.state_dim)
            state[:, : self.model.nq] = torch.tensor(self.scene.initial)
            # Velocity perturbations preserve joint constraints and object geometry.
            state[:, self.model.nq :] = (
                0.01
                * torch.randn(
                    self.batch_size, self.model.nv, generator=generator, device=getattr(generator, "device", "cpu")
                ).cpu()
            )
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state.shape != (self.batch_size, self.state_dim) or not torch.isfinite(state).all():
            raise ValueError("Invalid simulator state")
        self._reset_episode()
        for data, row in zip(self.data, state.detach().cpu().numpy(), strict=True):
            self._restore(data, row)
        self._sync()
        return self.render()

    def _sync(self):
        self.state = torch.as_tensor(
            self._np.stack([self._np.concatenate((d.qpos, d.qvel)) for d in self.data]),
            dtype=torch.float32,
            device=self.device,
        )

    def distance(self, states, goal):
        goal = goal.view(len(goal), *([1] * (states.ndim - 2)), -1)
        delta = states[..., self._indices] - goal[..., self._indices]
        delta = torch.where(self._angular, torch.atan2(delta.sin(), delta.cos()), delta)
        return (delta / self._tolerance).square().mean(-1).sqrt()

    def success(self, states, goal):
        settled = states[..., self.model.nq + self._velocity_indices].abs().amax(-1) < self.scene.velocity_tolerance
        bounded = states[..., self._bound_indices]
        released = ((bounded >= self._bound_lower) & (bounded <= self._bound_upper)).all(-1)
        return (self.distance(states, goal) < 1) & settled & released

    def step(self, action):
        action = self._action(action)
        active = ~self.transition.done
        controls = action.detach().cpu().numpy() * self._scale + self._mid
        for i in active.nonzero().flatten().cpu().tolist():
            self.data[i].ctrl[:] = controls[i]
            self.mj.mj_step(self.model, self.data[i], nstep=self.substeps)
        self._sync()
        self.elapsed += active
        reward = (-self.distance(self.state, self.goal) - 0.001 * action.square().sum(-1)) * active
        self.transition = Transition(
            reward,
            torch.zeros_like(active),
            self.elapsed >= self.max_episode_steps,
            self.success(self.state, self.goal),
        )
        return self.render()

    def render(self, state=None):
        state = self.state if state is None else torch.as_tensor(state, device=self.device)
        if self.observation == "state":
            return state.clone()
        return self._render_rgb(state, self.size)

    def _render_rgb(self, state, size, *, follow=False):
        if size not in self._renderers:
            self._renderers[size] = self.mj.Renderer(self.model, height=size, width=size)
        renderer = self._renderers[size]
        images = []
        if self.goal_marker and len(state) != self.batch_size:
            raise ValueError("Marked rendering requires one state per task target")
        for index, row in enumerate(state.detach().cpu().numpy()):
            self._restore(self._scratch, row)
            camera = self._camera
            if follow:
                lower, upper = self._geometry_bounds()
                lower = self._np.minimum(lower, self._scene_bounds[0])
                upper = self._np.maximum(upper, self._scene_bounds[1])
                camera = self._presentation_camera
                self._fit_camera(camera, lower, upper)
            renderer.update_scene(self._scratch, camera=camera)
            if self.goal_marker:
                target = self.goal if self.visual_goal is None else self.visual_goal
                self._restore(self._target_scratch, target[index].detach().cpu().numpy())
                # ObstacleArm's last body is the forearm; its endpoint is 0.35 m along local x.
                pose = self._target_scratch
                position = pose.xpos[-1] + pose.xmat[-1].reshape(3, 3) @ self._np.array([0.35, 0, 0])
                slot = renderer.scene.ngeom
                if slot >= renderer.scene.maxgeom:
                    raise RuntimeError("Renderer has no space for the task target")
                self.mj.mjv_initGeom(
                    renderer.scene.geoms[slot],
                    self.mj.mjtGeom.mjGEOM_SPHERE,
                    self._np.array([0.04, 0.04, 0.04]),
                    position,
                    self._np.eye(3).ravel(),
                    self._np.array([0.8, 0.2, 0.2, 0.8]),
                )
                renderer.scene.ngeom += 1
            renderer.scene.flags[self.mj.mjtRndFlag.mjRND_SHADOW] = self.render_quality.shadows
            images.append(renderer.render().copy())
        return torch.as_tensor(self._np.stack(images), device=self.device).permute(0, 3, 1, 2).float() / 255

    def render_frame(self, state=None):
        state = self.state if state is None else torch.as_tensor(state, device=self.device)
        return self._render_rgb(state, self.render_size, follow=self.scene.follow_camera)

    def render_goal_frame(self):
        return self.render_frame(self.goal)

    def render_target(self):
        if self.observation == "rgb":
            return self.render(self.goal)
        target = torch.full_like(self.goal, float("nan"))
        target[:, self._indices] = self.goal[:, self._indices]
        target[:, self._bound_indices] = self.goal[:, self._bound_indices]
        target[:, self.model.nq + self._velocity_indices] = 0
        return target

    def close(self):
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
