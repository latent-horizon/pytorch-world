from torch_world.envs.macaw import Macaw
from torch_world.envs.obstacle_arm import ObstacleArm
from torch_world.envs.point_mass import PointMass
from torch_world.envs.utils import run_policy

__all__ = ["Macaw", "ObstacleArm", "PointMass", "PushT", "RobotCube", "run_policy"]


def __getattr__(name):
    if name == "RobotCube":  # the MuJoCo environments need the optional ``mujoco`` dependency
        from torch_world.envs.robot_cube import RobotCube

        return RobotCube
    if name == "PushT":
        from torch_world.envs.push_t import PushT

        return PushT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
