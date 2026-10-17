"""State-feedback demonstration controllers for our three Franka tasks.

These privileged-state controllers are exclusively reference examples. Model
experiments continue to use their supplied model and planner factories.
"""

import torch


class FrankaReference:
    """Damped inverse-kinematics control through the environment's normal actions."""

    def __init__(self, world):
        from isaaclab.utils.math import quat_conjugate, quat_mul

        self.world = world
        self.robot = world.env.scene["robot"]
        self.hand = self.robot.find_bodies("panda_hand")[0][0]
        limits = self.robot.data.joint_pos_limits.clone()[:, :7]
        self.mid, self.scale = limits.mean(-1), (limits[..., 1] - limits[..., 0]) / 2
        self.command = self.robot.data.joint_pos.clone()[:, :7]
        self.orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=world.device).expand(world.batch_size, -1)
        self.quat_conjugate, self.quat_mul = quat_conjugate, quat_mul
        self.waypoint = {"reach": self._reach, "lift": self._lift, "push": self._push}[world.task]
        self.integrate = world.task != "lift"
        self.increment = 0.01 if world.task == "push" else 0.025

    def _point(self, values):
        return torch.tensor(values, device=self.world.device).expand(self.world.batch_size, -1)

    def _reach(self, time):
        from torch_world.envs.isaac_scenes import TARGETS

        return self._point(TARGETS["reach"]), 1.0

    def _lift(self, time):
        if time < 2:
            return self._point([0.5, -0.12, 0.25]), 1.0
        if time < 4:
            return self._point([0.5, -0.12, 0.135]), 1.0
        if time < 5.5:
            return self._point([0.5, -0.12, 0.135]), -1.0
        # Lift gradually: a discontinuous Cartesian jump can eject a contact grasp.
        return self._point([0.54, min(0, -0.12 + (time - 5.5) * 0.04), min(0.44, 0.135 + (time - 5.5) * 0.05)]), -1.0

    def _push(self, time):
        from torch_world.envs.isaac_scenes import TARGETS

        cube = self.world.state[:, -6:-3]
        direction = self._point(TARGETS["push"])[:, :2] - cube[:, :2]
        direction = torch.nn.functional.normalize(direction, dim=-1)
        target = cube.clone()
        target[:, :2] -= (0.10 if time < 4 else 0.035) * direction
        target[:, 2] = 0.3 if time < 2 else 0.145
        return target, -1.0

    @torch.no_grad()
    def __call__(self, observation):
        position = self.robot.data.body_pos_w[:, self.hand].clone() - self.world.env.scene.env_origins
        joints = self.robot.data.joint_pos.clone()[:, :7]
        time = int(self.world.elapsed[0]) * self.world.env.step_dt
        target, grip = self.waypoint(time)
        error = target - position
        jacobian = self.robot.data.body_link_jacobian_w[:, self.hand, :, :7].clone()
        if self.world.task == "reach":
            jacobian = jacobian[:, :3]
        else:
            quaternion = self.robot.data.body_quat_w[:, self.hand].clone()
            delta = self.quat_mul(self.orientation, self.quat_conjugate(quaternion))
            rotation = 2 * delta[:, :3] * delta[:, 3:].sign()
            error = torch.cat((error, 0.4 * rotation), -1)
            jacobian[:, 3:] *= 0.4
        regularizer = 0.002 * torch.eye(jacobian.shape[1], device=self.world.device)
        delta = (
            jacobian.transpose(-1, -2)
            @ torch.linalg.solve(jacobian @ jacobian.transpose(-1, -2) + regularizer, error.unsqueeze(-1))
        ).squeeze(-1)
        self.command = (
            self.command + (0.1 * delta).clamp(-self.increment, self.increment)
            if self.integrate
            else joints + (0.5 * delta).clamp(-0.04, 0.04)
        )
        self.command = self.command.clamp(self.mid - self.scale, self.mid + self.scale)
        arm = (self.command - self.mid) / self.scale
        gripper = torch.full((self.world.batch_size, 2), grip, device=self.world.device)
        return torch.cat((arm, gripper), -1).clamp(-1, 1)
