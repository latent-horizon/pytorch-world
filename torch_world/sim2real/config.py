"""Serializable configuration shared by simulation and physical SO101 runs.

Lengths are metres, angles are radians, and times are seconds unless a field
explicitly names another unit. Example geometry is not a hardware calibration.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
PIECES = ("pawn", "rook", "knight", "bishop", "queen", "king")


@dataclass(frozen=True)
class BoardAppearance:
    """Wooden case appearance. Dimensions are estimates until the task is measured."""

    border: float = 0.014
    thickness: float = 0.002
    surface: float = 0.003
    light: tuple[float, float, float] = (0.83, 0.73, 0.60)
    dark: tuple[float, float, float] = (0.38, 0.17, 0.10)
    frame: tuple[float, float, float] = (0.48, 0.23, 0.13)
    grain: float = 0.12

    def __post_init__(self):
        import math

        if (
            not all(math.isfinite(v) and v >= 0 for v in (self.border, self.thickness, self.grain))
            or not math.isfinite(self.surface)
            or not self.surface > 0
        ):
            raise ValueError("Invalid board appearance dimensions")
        for color in (self.light, self.dark, self.frame):
            if len(color) != 3 or not all(math.isfinite(v) and 0 <= v <= 1 for v in color):
                raise ValueError("Board colors must be RGB values in [0,1]")


@dataclass(frozen=True)
class WristView:
    """Approximate camera mount in gripperframe coordinates; not a calibration."""

    eye: tuple[float, float, float] = (-0.06, 0.0, 0.035)
    target: tuple[float, float, float] = (0.04, 0.0, 0.0)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fovy: float = 75.0
    aspect: float = 1.0

    def __post_init__(self):
        import math

        if not all(len(v) == 3 and all(math.isfinite(x) for x in v) for v in (self.eye, self.target, self.up)):
            raise ValueError("Camera vectors must contain three finite values")
        if self.eye == self.target or not 1 < self.fovy < 179 or not 0.5 <= self.aspect <= 2:
            raise ValueError("Invalid camera projection")


@dataclass(frozen=True)
class PieceSpec:
    """Measured outer dimensions and grasp height, relative to the piece base."""

    height: float = 0.04
    diameter: float = 0.022
    neck_diameter: float = 0.012
    grasp_height: float = 0.019
    mass: float = 0.012
    friction: float = 1.2

    def __post_init__(self):
        import math

        if not all(math.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError("Piece measurements must be finite and positive")
        if not self.neck_diameter <= self.diameter or not self.grasp_height < self.height:
            raise ValueError("Invalid piece neck diameter or grasp height")


def _pieces():
    return {
        name: PieceSpec(height=h, grasp_height=g, mass=m)
        for name, h, g, m in zip(
            PIECES,
            (0.04, 0.042, 0.05, 0.052, 0.06, 0.065),
            (0.019, 0.02, 0.023, 0.025, 0.027, 0.029),
            (0.012, 0.015, 0.016, 0.017, 0.02, 0.022),
        )
    }


@dataclass(frozen=True)
class ChessTaskConfig:
    """An empty 8x8 board; only reachable squares are sampled.

    ``board_origin`` is the centre of a1 in the robot base frame. Files run
    along board +x and ranks along board +y, rotated by ``board_yaw``.
    """

    square_size: float = 0.032
    board_origin: tuple[float, float, float] = (0.14, -0.112, 0.003)
    board_yaw: float = 0.0
    appearance: BoardAppearance = field(default_factory=BoardAppearance)
    pieces: dict[str, PieceSpec] = field(default_factory=_pieces)
    placement_tolerance: float = 0.005
    upright_tolerance: float = 0.15
    settle_steps: int = 5
    max_steps: int = 240
    hover_height: float = 0.095
    measured: bool = False

    def __post_init__(self):
        import math

        values = (
            *self.board_origin,
            self.board_yaw,
            self.square_size,
            self.placement_tolerance,
            self.upright_tolerance,
            self.hover_height,
        )
        if not all(math.isfinite(v) for v in values) or self.square_size <= 0 or self.hover_height <= 0:
            raise ValueError("Invalid board geometry")
        if not 0 < self.placement_tolerance < self.square_size / 2 or not 0 < self.upright_tolerance < 1:
            raise ValueError("Invalid placement tolerance")
        if self.settle_steps < 1 or self.max_steps < 1 or set(self.pieces) != set(PIECES):
            raise ValueError("Specify all six piece types and positive episode limits")
        if any(p.diameter >= self.square_size for p in self.pieces.values()):
            raise ValueError("Piece bases must fit inside a square")


@dataclass(frozen=True)
class SO101Config:
    """Shared camera/control contract; approach pitch is below the horizontal radial plane."""

    control_dt: float = 0.1
    max_step: float = 0.005
    max_joint_step: float = 0.15
    max_gripper_step: float = 0.15
    size: int = 84
    arm_color: tuple[float, float, float] = (0.92, 0.93, 0.94)
    wrist_view: WristView = field(default_factory=WristView)
    approach_pitch: float = 1.5707963267948966
    workspace_low: tuple[float, float, float] = (0.12, -0.14, 0.006)
    workspace_high: tuple[float, float, float] = (0.32, 0.14, 0.18)
    max_observation_age: float = 0.5
    ik_tolerance: float = 0.004
    seed: int = 7

    def __post_init__(self):
        import math

        positive = (
            self.control_dt,
            self.max_step,
            self.max_joint_step,
            self.max_gripper_step,
            self.max_observation_age,
            self.ik_tolerance,
        )
        if not all(math.isfinite(v) and v > 0 for v in positive) or self.size < 16:
            raise ValueError("Invalid controller/image configuration")
        if len(self.arm_color) != 3 or not all(math.isfinite(v) and 0 <= v <= 1 for v in self.arm_color):
            raise ValueError("arm_color must contain three finite RGB values in [0,1]")
        if not math.isfinite(self.approach_pitch) or not 0.5 <= self.approach_pitch <= math.pi / 2:
            raise ValueError("approach_pitch must be between 0.5 and pi/2 radians below horizontal")
        if (
            len(self.workspace_low) != 3
            or len(self.workspace_high) != 3
            or not all(
                math.isfinite(a) and math.isfinite(b) and a < b for a, b in zip(self.workspace_low, self.workspace_high)
            )
        ):
            raise ValueError("Invalid workspace bounds")


@dataclass(frozen=True)
class RandomizationConfig:
    """Episode-wise ranges around the calibrated nominal scene."""

    enabled: bool = True
    dimension_scale: tuple[float, float] = (0.95, 1.05)
    mass_scale: tuple[float, float] = (0.8, 1.2)
    friction_scale: tuple[float, float] = (0.8, 1.2)
    gain_scale: tuple[float, float] = (0.85, 1.15)
    light_scale: tuple[float, float] = (0.7, 1.2)
    camera_translation: float = 0.002
    camera_rotation: float = 0.02
    pixel_noise: float = 0.015
    max_action_delay: int = 1
    max_observation_delay: int = 1

    def __post_init__(self):
        import math

        for key in ("dimension_scale", "mass_scale", "friction_scale", "gain_scale", "light_scale"):
            lo, hi = getattr(self, key)
            if not math.isfinite(lo + hi) or not 0 < lo <= hi:
                raise ValueError(f"Invalid randomization range: {key}")
        for key in ("camera_translation", "camera_rotation", "pixel_noise"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"Invalid randomization magnitude: {key}")
        if self.max_action_delay < 0 or self.max_observation_delay < 0:
            raise ValueError("Delays must be nonnegative")


def fingerprint(*configs):
    """Stable identifier for action, preprocessing, task and calibration contracts."""
    values = []
    for config in configs:
        value = asdict(config) if hasattr(config, "__dataclass_fields__") else config
        # Preserve contracts written before configurable approach pitch existed.
        # A nondefault tilt is always bound into new checkpoints and goals.
        if isinstance(config, SO101Config) and config.approach_pitch == SO101Config.approach_pitch:
            value.pop("approach_pitch")
        values.append(value)
    return hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_config(path, **configs):
    Path(path).write_text(json.dumps({k: asdict(v) for k, v in configs.items()}, indent=2, allow_nan=False) + "\n")
