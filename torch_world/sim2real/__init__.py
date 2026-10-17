"""SO101 chess simulation, world-model experiments and explicit LeRobot deployment.

MuJoCo and LeRobot are imported only when constructing the corresponding backend.
"""

from torch_world.sim2real.calibration import (
    Calibration,
    board_pose,
    calibrate_camera,
    calibrate_hand_eye,
    fit_joint_mapping,
    overlay,
)
from torch_world.sim2real.camera import capture_wrist_camera
from torch_world.sim2real.config import (
    BoardAppearance,
    ChessTaskConfig,
    PieceSpec,
    RandomizationConfig,
    SO101Config,
    WristView,
    save_config,
)
from torch_world.sim2real.control import Observation
from torch_world.sim2real.data import Episode, EpisodeWindows, LeWAMWindows, collect_episodes
from torch_world.sim2real.experiment import (
    evaluate,
    label_trial,
    load_checkpoint,
    run_experiment,
    run_trial,
    save_checkpoint,
    scripted_episode,
    train,
)
from torch_world.sim2real.hardware import MockDriver, SO101Robot, load_goal, save_goal
from torch_world.sim2real.models import ModelAdapter, WorldModelPolicy, make_adapter
from torch_world.sim2real.reference import reference_setup, render_reference_pose, simulation_gate
from torch_world.sim2real.simulation import SO101Chess

__all__ = [
    "BoardAppearance",
    "Calibration",
    "ChessTaskConfig",
    "Episode",
    "EpisodeWindows",
    "LeWAMWindows",
    "MockDriver",
    "ModelAdapter",
    "Observation",
    "PieceSpec",
    "RandomizationConfig",
    "SO101Chess",
    "SO101Config",
    "SO101Robot",
    "WorldModelPolicy",
    "WristView",
    "board_pose",
    "calibrate_camera",
    "calibrate_hand_eye",
    "capture_wrist_camera",
    "collect_episodes",
    "evaluate",
    "fit_joint_mapping",
    "label_trial",
    "load_checkpoint",
    "load_goal",
    "make_adapter",
    "overlay",
    "reference_setup",
    "render_reference_pose",
    "run_experiment",
    "run_trial",
    "save_checkpoint",
    "save_config",
    "save_goal",
    "scripted_episode",
    "simulation_gate",
    "train",
]
