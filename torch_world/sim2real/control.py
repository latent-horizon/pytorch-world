"""The same Cartesian controller and SO101 kinematic model on both backends."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from torch_world.envs.mujoco import import_mujoco
from torch_world.envs.robot_cube import ROBOTS, _build
from torch_world.sim2real.config import JOINTS, SO101Config


@dataclass
class Observation:
    """Single-robot observation; RGB is (3,H,W) float32 in [0,1].

    ``joints`` contains five measured arm angles in radians; ``opening`` is
    measured gripper opening in [0,1]. Timestamp is monotonic acquisition time
    (simulation time on the simulator), never the time a cached frame is read.
    """

    rgb: torch.Tensor
    joints: torch.Tensor
    opening: float
    timestamp: float
    contract: str | None = None
    commanded: torch.Tensor | None = None

    @property
    def proprio(self):
        return torch.cat((self.joints, self.joints.new_tensor([self.opening])))

    def state(self, include_commanded=False):
        """Measured state, optionally followed by commanded XYZ/opening."""
        if not include_commanded:
            return self.proprio
        if self.commanded is None or self.commanded.shape != (4,) or not torch.isfinite(self.commanded).all():
            raise ValueError("This policy requires a recorded commanded XYZ/opening")
        return torch.cat((self.proprio, self.commanded.to(self.joints)))


@dataclass
class Command:
    requested: np.ndarray
    applied: np.ndarray
    joints: np.ndarray
    opening: float
    target: np.ndarray


class ClearanceError(ValueError):
    """A rejected kinematic path, with contact evidence; no physics step occurred."""

    def __init__(self, contacts, start, target, opening):
        super().__init__("Joint path penetrates the table or board")
        self.contacts = contacts
        self.start, self.target = np.asarray(start).copy(), np.asarray(target).copy()
        self.opening = float(opening)


class SO101Controller:
    """Damped least-squares IK, workspace limits and joint-rate limiting.

    Controller targets are initialized from measurement, never from a simulator
    home pose. FK uses a private scratch MjData and cannot advance a live world.
    """

    def __init__(self, config: SO101Config, model=None, root=None):
        self.config = config
        self.mj = import_mujoco()
        self.model = (
            model if model is not None else _build("so101", Path(root or Path.home() / ".cache/torch_world/menagerie"))
        )
        self.data = self.mj.MjData(self.model)
        self.indices = np.array([self.model.joint(n).qposadr[0] for n in JOINTS])
        self.dofs = np.array([self.model.joint(n).dofadr[0] for n in JOINTS])
        self.limits = np.array([self.model.joint(n).range for n in JOINTS])
        self.actuators = np.array([self.model.actuator(n).id for n in JOINTS])
        self.site = self.model.site("grasp").id
        self.data.qpos[self.indices] = ROBOTS["so101"].home
        self.mj.mj_forward(self.model, self.data)
        self.down = self.data.site_xmat[self.site].reshape(3, 3)[:, 0].copy()
        self.pan_origin = self.data.xanchor[self.model.joint("shoulder_pan").id].copy()
        self.target = None
        self.opening = 1.0

    def pose(self, joints):
        self.data.qpos[self.indices] = np.asarray(joints)
        self.mj.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.site].copy()

    def reset(self, joints, opening):
        self.target = self.pose(joints)
        self.opening = float(opening)

    def solve(self, joints, target, iterations=100):
        q = np.asarray(joints, dtype=float).copy()
        target = np.asarray(target, dtype=float)
        # A nonvertical approach follows the arm's radial plane. A fixed world
        # tilt would overconstrain this five-axis arm on off-centre squares.
        desired_axis = self.down
        if self.config.approach_pitch < np.pi / 2 - 1e-6:
            yaw = np.arctan2(target[1] - self.pan_origin[1], target[0] - self.pan_origin[0])
            pitch = self.config.approach_pitch
            desired_axis = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)])
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(iterations):
            position = self.pose(q)
            axis = self.data.site_xmat[self.site].reshape(3, 3)[:, 0]
            error = np.r_[target - position, 0.5 * np.cross(axis, desired_axis)]
            self.mj.mj_jacSite(self.model, self.data, jacp, jacr, self.site)
            jac = np.concatenate((jacp, jacr))[:, self.dofs]
            q = np.clip(
                q + jac.T @ np.linalg.solve(jac @ jac.T + 1e-3 * np.eye(6), error), self.limits[:, 0], self.limits[:, 1]
            )
        residual = np.linalg.norm(self.pose(q) - target)
        return q, residual

    def clearance_contacts(self, start, target, opening, *, start_opening=None):
        """Inspect the five sampled path poses without advancing the live world.

        By default this reproduces the controller's conservative check, using
        the requested opening throughout. ``start_opening`` is a diagnostic
        alternative that interpolates the measured opening along the path.
        Neither check certifies continuous swept-volume collision freedom.
        """
        base = self.model.body("base").id
        ignored = {0, base}
        for name in ("cube", "piece", "goal"):
            try:
                ignored.add(self.model.body(name).id)
            except KeyError:
                pass
        contacts = []
        start_opening = opening if start_opening is None else start_opening
        for fraction in np.linspace(0, 1, 5):
            checked_opening = start_opening + fraction * (opening - start_opening)
            self.data.qpos[self.model.joint("gripper").qposadr[0]] = -0.1 + 1.6 * checked_opening
            self.pose(np.asarray(start) + fraction * (np.asarray(target) - start))
            for contact in self.data.contact:
                bodies = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
                if contact.dist < -0.001 and 0 in bodies and any(b not in ignored for b in bodies):
                    contacts.append(
                        {
                            "fraction": float(fraction),
                            "opening": float(checked_opening),
                            "distance_m": float(contact.dist),
                            "geom_ids": [int(contact.geom1), int(contact.geom2)],
                            "geoms": [self.model.geom(g).name for g in (contact.geom1, contact.geom2)],
                            "bodies": [self.model.body(b).name for b in bodies],
                        }
                    )
        return contacts

    def check_clearance(self, start, target, opening):
        """Reject sampled joint paths penetrating the table/board with moving links."""
        contacts = self.clearance_contacts(start, target, opening)
        if contacts:
            raise ClearanceError(contacts, start, target, opening)

    def command(self, action, joints, opening):
        a = np.asarray(torch.as_tensor(action).detach().cpu(), dtype=float)
        q = np.asarray(joints, dtype=float)
        if a.shape != (4,) or q.shape != (5,) or not np.isfinite(np.r_[a, q, opening]).all():
            raise ValueError("Expected finite action (4,), joints (5,) and opening")
        if np.any(q < self.limits[:, 0] - 0.02) or np.any(q > self.limits[:, 1] + 0.02):
            raise ValueError("Measured joints outside the simulator/calibrated limits")
        if self.target is None:
            self.reset(q, opening)
        bounded = np.clip(a, -1, 1)
        target = np.clip(
            self.target + self.config.max_step * bounded[:3], self.config.workspace_low, self.config.workspace_high
        )
        solved, residual = self.solve(q, target)
        if residual > self.config.ik_tolerance:
            raise ValueError(f"Unreachable Cartesian target (IK residual {residual:.4f} m)")
        limited = np.clip(solved, q - self.config.max_joint_step, q + self.config.max_joint_step)
        # Reconcile rate-limited commands; do not let an unreachable target accumulate.
        if not np.allclose(limited, solved):
            target = self.pose(limited)
        new_opening = float(
            np.clip(
                self.opening + np.clip(bounded[3], -self.config.max_gripper_step, self.config.max_gripper_step), 0, 1
            )
        )
        self.check_clearance(q, limited, new_opening)
        applied = np.r_[(target - self.target) / self.config.max_step, new_opening - self.opening]
        self.target, self.opening = target.copy(), new_opening
        return Command(a.copy(), applied, limited, new_opening, target)

    def reconcile(self, joints, opening):
        """Use the driver's returned command if it applied additional limiting."""
        self.target = self.pose(joints)
        self.opening = float(opening)
