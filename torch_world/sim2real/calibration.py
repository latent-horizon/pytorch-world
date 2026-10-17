"""Explicit motor and camera calibration, independent of LeRobot imports."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch_world.sim2real.config import JOINTS, fingerprint


def rigid_transform(value):
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Expected a finite 4x4 transform")
    rotation = transform[:3, :3]
    if not (
        np.allclose(transform[3], [0, 0, 0, 1])
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        and np.isclose(np.linalg.det(rotation), 1, atol=1e-5)
    ):
        raise ValueError("Expected a proper rigid transform")
    return transform


@dataclass(frozen=True)
class Calibration:
    """LeRobot degrees = signs * degrees(sim radians) + offsets_degrees.

    ``wrist_from_camera`` maps OpenCV camera coordinates (+x right, +y down,
    +z forward) into the Menagerie ``gripperframe`` site. ``intrinsics`` are
    (fx,fy,cx,cy) for ``resolution`` (width,height), before resizing. Gripper
    endpoints are in LeRobot's separate 0..100 range, even with use_degrees=True.
    """

    robot_id: str
    signs: tuple[float, ...]
    offsets_degrees: tuple[float, ...]
    gripper_closed: float
    gripper_open: float
    intrinsics: tuple[float, float, float, float]
    resolution: tuple[int, int]
    wrist_from_camera: tuple[tuple[float, ...], ...]
    distortion: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    verified: bool = False

    def __post_init__(self):
        if not self.robot_id or len(self.signs) != 5 or len(self.offsets_degrees) != 5:
            raise ValueError("A robot id and five joint mappings are required")
        if not set(self.signs) <= {-1, 1} or not np.isfinite(self.offsets_degrees).all():
            raise ValueError("Joint signs must be +/-1 and offsets finite")
        if not 0 <= self.gripper_closed <= 100 or not 0 <= self.gripper_open <= 100:
            raise ValueError("Gripper endpoints must be in LeRobot's 0..100 range")
        if abs(self.gripper_open - self.gripper_closed) < 1:
            raise ValueError("Gripper endpoints must differ")
        if len(self.intrinsics) != 4 or not np.isfinite(self.intrinsics).all() or min(self.intrinsics[:2]) <= 0:
            raise ValueError("Invalid camera intrinsics")
        if len(self.resolution) != 2 or min(self.resolution) <= 0:
            raise ValueError("Invalid camera resolution")
        if len(self.distortion) not in (4, 5, 8, 12, 14) or not np.isfinite(self.distortion).all():
            raise ValueError("Invalid OpenCV distortion coefficients")
        rigid_transform(self.wrist_from_camera)

    @property
    def id(self):
        return fingerprint(self)

    def to_driver(self, joints, opening):
        q = np.asarray(joints, dtype=float)
        if q.shape != (5,) or not np.isfinite(q).all() or not np.isfinite(opening) or not 0 <= opening <= 1:
            raise ValueError("Invalid joint/gripper command")
        values = np.asarray(self.signs) * np.rad2deg(q) + self.offsets_degrees
        return {
            **{f"{name}.pos": float(v) for name, v in zip(JOINTS, values)},
            "gripper.pos": self.gripper_closed + opening * (self.gripper_open - self.gripper_closed),
        }

    def from_driver(self, values):
        degrees = np.array([values[f"{name}.pos"] for name in JOINTS])
        opening = (values["gripper.pos"] - self.gripper_closed) / (self.gripper_open - self.gripper_closed)
        if not np.isfinite(np.r_[degrees, opening]).all() or not -0.05 <= opening <= 1.05:
            raise ValueError("Invalid measured driver positions")
        return np.deg2rad((degrees - self.offsets_degrees) * self.signs), float(np.clip(opening, 0, 1))

    def preprocess(self, rgb, size):
        """Undistort RGB at native resolution, then resize identically for all models."""
        frame = np.asarray(rgb)
        if frame.shape != (self.resolution[1], self.resolution[0], 3) or frame.dtype != np.uint8:
            raise ValueError("Camera must supply uint8 RGB at the calibrated resolution")
        fx, fy, cx, cy = self.intrinsics
        matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        if np.any(self.distortion):
            import cv2

            frame = cv2.undistort(frame, matrix, np.asarray(self.distortion), None, matrix)
        tensor = torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255
        return F.interpolate(tensor[None], (size, size), mode="bilinear", align_corners=False, antialias=True)[0]

    def apply_camera(self, model, name):
        from torch_world.envs.mujoco import import_mujoco

        mj = import_mujoco()
        site = model.site("gripperframe")
        rotation = np.zeros(9)
        mj.mju_quat2Mat(rotation, site.quat)
        body_from_wrist = np.eye(4)
        body_from_wrist[:3, :3], body_from_wrist[:3, 3] = rotation.reshape(3, 3), site.pos
        camera = model.camera(name)
        if camera.bodyid[0] != site.bodyid[0]:
            raise ValueError("Wrist camera must be attached to the gripperframe body")
        body_from_camera = body_from_wrist @ rigid_transform(self.wrist_from_camera)
        camera.pos[:] = body_from_camera[:3, 3]
        # OpenCV looks along +z with y down; MuJoCo looks along -z with y up.
        rotation = body_from_camera[:3, :3] @ np.diag([1.0, -1.0, -1.0])
        mj.mju_mat2Quat(camera.quat, rotation.flatten())
        width, height = self.resolution
        fx, fy, cx, cy = self.intrinsics
        model.cam_resolution[camera.id] = [width, height]
        model.cam_sensorsize[camera.id] = [1, 1]
        model.cam_intrinsic[camera.id] = [fx / width, fy / height, (cx - width / 2) / width, (height / 2 - cy) / height]

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    @classmethod
    def from_simulator(cls, model, *, resolution=(640, 480), robot_id="simulation-only"):
        """Unverified example profile for mock runs, never a physical calibration."""
        from torch_world.envs.mujoco import import_mujoco

        mj = import_mujoco()
        site, camera = model.site("gripperframe"), model.camera("wrist")
        transforms = []
        for pos, quat in ((site.pos, site.quat), (camera.pos, camera.quat)):
            rotation = np.zeros(9)
            mj.mju_quat2Mat(rotation, quat)
            transform = np.eye(4)
            transform[:3, :3], transform[:3, 3] = rotation.reshape(3, 3), pos
            transforms.append(transform)
        wrist_from_camera = np.linalg.inv(transforms[0]) @ transforms[1] @ np.diag([1.0, -1.0, -1.0, 1.0])
        width, height = resolution
        fy = height / (2 * np.tan(np.deg2rad(float(camera.fovy[0])) / 2))
        return cls(
            robot_id,
            (1,) * 5,
            (0.0,) * 5,
            0.0,
            100.0,
            (fy, fy, width / 2, height / 2),
            resolution,
            tuple(tuple(row) for row in wrist_from_camera),
        )


def fit_joint_mapping(sim_radians, driver_degrees, *, max_error_degrees=2.0):
    """Fit signs and offsets from >=3 manually matched reference poses.

    Each joint must span at least 10 degrees. Returns mapping and per-joint RMS
    error; rejects an inconsistent mapping rather than fitting an arbitrary scale.
    """
    sim, real = np.rad2deg(np.asarray(sim_radians)), np.asarray(driver_degrees)
    if sim.ndim != 2 or sim.shape[1] != 5 or sim.shape[0] < 3 or sim.shape != real.shape:
        raise ValueError("Expected matching (N>=3,5) pose arrays")
    if not np.isfinite(sim).all() or not np.isfinite(real).all() or np.any(np.ptp(sim, axis=0) < 10):
        raise ValueError("Reference poses must be finite and span each joint")
    signs = np.where(((sim - sim.mean(0)) * (real - real.mean(0))).sum(0) >= 0, 1.0, -1.0)
    offsets = (real - signs * sim).mean(0)
    errors = np.sqrt(((real - (signs * sim + offsets)) ** 2).mean(0))
    if np.any(errors > max_error_degrees):
        raise ValueError(f"Joint mapping residual too large: {errors}")
    return {"signs": signs.tolist(), "offsets_degrees": offsets.tolist(), "rms_degrees": errors.tolist()}


def calibrate_camera(object_points, image_points, resolution):
    """OpenCV intrinsic fit from corresponding checkerboard points in >=5 views."""
    import cv2

    if len(object_points) < 5 or len(image_points) != len(object_points):
        raise ValueError("At least five matching calibration views are required")
    error, matrix, distortion, _, _ = cv2.calibrateCamera(
        [np.asarray(p, np.float32) for p in object_points],
        [np.asarray(p, np.float32) for p in image_points],
        tuple(resolution),
        None,
        None,
    )
    return {
        "intrinsics": [matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]],
        "distortion": distortion.ravel().tolist(),
        "rms_pixels": float(error),
    }


def calibrate_hand_eye(base_from_wrist, camera_from_board):
    """Eye-in-hand calibration from >=5 diverse poses of a stationary board.

    Inputs are paired 4x4 transforms. Returns wrist_from_camera (OpenCV axes).
    Rotations about multiple axes are required for an observable calibration.
    """
    import cv2

    if len(base_from_wrist) < 5 or len(base_from_wrist) != len(camera_from_board):
        raise ValueError("At least five paired, diverse calibration poses are required")
    a, b = [rigid_transform(t) for t in base_from_wrist], [rigid_transform(t) for t in camera_from_board]
    r, t = cv2.calibrateHandEye(
        [v[:3, :3] for v in a], [v[:3, 3] for v in a], [v[:3, :3] for v in b], [v[:3, 3] for v in b]
    )
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = r, t.ravel()
    return rigid_transform(result)


def board_pose(base_from_wrist, wrist_from_camera, camera_from_board):
    """Compose measured board-to-base alignment; no motion or pose estimation side effects."""
    return rigid_transform(base_from_wrist) @ rigid_transform(wrist_from_camera) @ rigid_transform(camera_from_board)


def overlay(observed, simulated, alpha=0.5):
    """Blend aligned CHW RGB tensors for visual calibration inspection."""
    if observed.shape != simulated.shape or not 0 <= alpha <= 1:
        raise ValueError("Overlay requires equal image shapes and alpha in [0,1]")
    return observed * (1 - alpha) + simulated.to(observed) * alpha
