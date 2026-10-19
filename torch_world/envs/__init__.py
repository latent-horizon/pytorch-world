from typing import TYPE_CHECKING

from torch_world.envs.benchmark import BenchmarkEnv, Transition
from torch_world.envs.catalog import CATEGORIES, register_builtins
from torch_world.envs.macaw import Macaw
from torch_world.envs.physics_scenes import Scene
from torch_world.envs.point_mass import PointMass
from torch_world.envs.registry import EnvSpec, get_spec, list_envs, make, register_env, register_task
from torch_world.envs.rendering import QUALITY_PRESETS, RenderQuality
from torch_world.envs.utils import run_policy

if TYPE_CHECKING:
    from torch_world.envs.tasks import (  # noqa: F401
        BallPlate,
        BlockStack,
        BoxGate,
        CartpoleSwingup,
        Checkpoints,
        DifferentialParking,
        DoorOpen,
        DoubleCartpole,
        DrawerOpen,
        FrankaCubeLift,
        FrankaObstacleReach,
        FrankaTablePush,
        HurdleHopper,
        IceMaze,
        KeyDoor,
        Maze,
        NewtonAcrobot,
        NewtonPendulum,
        ObstacleArm,
        PayloadTransport,
        PegInsertion,
        PickPlace,
        PlanarQuadrotor,
        PuckPush,
        PuckSlide,
        ReactionWheel,
        Sokoban,
        SpatialArm,
    )

__all__ = [
    "CATEGORIES",
    "QUALITY_PRESETS",
    "BenchmarkEnv",
    "DMControl",
    "EnvSpec",
    "Macaw",
    "ManiSkill",
    "MetaWorld",
    "MyoSuite",
    "OGBench",
    "PointMass",
    "PushT",
    "RenderQuality",
    "RoboCasa",
    "RobotCube",
    "Scene",
    "Transition",
    "get_spec",
    "list_envs",
    "make",
    "register_env",
    "register_task",
    "run_policy",
]

register_builtins()
del register_builtins

# External suites keep their own names under a namespace, such as `OGBench/...` or `RoboCasa/...`, which are not
# identifiers; the suite's class, such as `OGBench(name)` or `RoboCasa(task)`, builds them.
_TASK_NAMES = frozenset(
    spec.id.removesuffix("-v0") for spec in list_envs() if "legacy" not in spec.tags and "/" not in spec.id
)
__all__.extend(sorted(_TASK_NAMES))


def __getattr__(name):
    if name in _TASK_NAMES:
        from torch_world.envs import tasks

        cls = getattr(tasks, name)
        globals()[name] = cls
        return cls
    if name == "RobotCube":  # the MuJoCo environments need the optional dependency
        from torch_world.envs.robot_cube import RobotCube

        return RobotCube
    external = {"DMControl": "dmcontrol", "MetaWorld": "metaworld", "ManiSkill": "maniskill", "MyoSuite": "myosuite"}
    if name in external:
        from importlib import import_module

        return getattr(import_module(f"torch_world.envs.{external[name]}"), name)
    if name == "OGBench":
        from torch_world.envs.ogbench import OGBench

        return OGBench
    if name == "RoboCasa":
        from torch_world.envs.robocasa import RoboCasa

        return RoboCasa
    if name == "PushT":
        from torch_world.envs.push_t import PushT

        return PushT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
