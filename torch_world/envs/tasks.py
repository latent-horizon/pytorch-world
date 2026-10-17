"""Named task classes sharing the registered scenes and backend implementations.

Importing these classes does not import or initialize any optional physics engine.
Constructor defaults and experiment metadata come from the same registry recipe.
"""

from copy import deepcopy

from torch_world.envs.isaaclab import IsaacLabEnv
from torch_world.envs.newton import NewtonPendulum as _NewtonPendulum
from torch_world.envs.physics_tasks import MujocoTask
from torch_world.envs.puzzles import PlanningPuzzle
from torch_world.envs.registry import constructor_settings, get_spec


class _RegisteredTask:
    env_id: str

    def __init__(self, **kwargs):
        spec = get_spec(self.env_id)
        settings = deepcopy({**spec.kwargs, **kwargs})
        self.settings = constructor_settings(super().__init__, settings)
        super().__init__(**settings)
        self.spec = spec


class Maze(_RegisteredTask, PlanningPuzzle):
    env_id = "Maze-v0"


class KeyDoor(_RegisteredTask, PlanningPuzzle):
    env_id = "KeyDoor-v0"


class Sokoban(_RegisteredTask, PlanningPuzzle):
    env_id = "Sokoban-v0"


class IceMaze(_RegisteredTask, PlanningPuzzle):
    env_id = "IceMaze-v0"


class Checkpoints(_RegisteredTask, PlanningPuzzle):
    env_id = "Checkpoints-v0"


class CartpoleSwingup(_RegisteredTask, MujocoTask):
    env_id = "CartpoleSwingup-v0"


class DoubleCartpole(_RegisteredTask, MujocoTask):
    env_id = "DoubleCartpole-v0"


class ReactionWheel(_RegisteredTask, MujocoTask):
    env_id = "ReactionWheel-v0"


class BallPlate(_RegisteredTask, MujocoTask):
    env_id = "BallPlate-v0"


class PlanarQuadrotor(_RegisteredTask, MujocoTask):
    env_id = "PlanarQuadrotor-v0"


class PayloadTransport(_RegisteredTask, MujocoTask):
    env_id = "PayloadTransport-v0"


class DifferentialParking(_RegisteredTask, MujocoTask):
    env_id = "DifferentialParking-v0"


class HurdleHopper(_RegisteredTask, MujocoTask):
    env_id = "HurdleHopper-v0"


class ObstacleArm(_RegisteredTask, MujocoTask):
    env_id = "ObstacleArm-v0"


class SpatialArm(_RegisteredTask, MujocoTask):
    env_id = "SpatialArm-v0"


class PuckPush(_RegisteredTask, MujocoTask):
    env_id = "PuckPush-v0"


class PuckSlide(_RegisteredTask, MujocoTask):
    env_id = "PuckSlide-v0"


class BoxGate(_RegisteredTask, MujocoTask):
    env_id = "BoxGate-v0"


class PegInsertion(_RegisteredTask, MujocoTask):
    env_id = "PegInsertion-v0"


class DrawerOpen(_RegisteredTask, MujocoTask):
    env_id = "DrawerOpen-v0"


class DoorOpen(_RegisteredTask, MujocoTask):
    env_id = "DoorOpen-v0"


class PickPlace(_RegisteredTask, MujocoTask):
    env_id = "PickPlace-v0"


class BlockStack(_RegisteredTask, MujocoTask):
    env_id = "BlockStack-v0"


class NewtonPendulum(_RegisteredTask, _NewtonPendulum):
    env_id = "NewtonPendulum-v0"


class NewtonAcrobot(_RegisteredTask, _NewtonPendulum):
    env_id = "NewtonAcrobot-v0"


class FrankaObstacleReach(_RegisteredTask, IsaacLabEnv):
    env_id = "FrankaObstacleReach-v0"


class FrankaTablePush(_RegisteredTask, IsaacLabEnv):
    env_id = "FrankaTablePush-v0"


class FrankaCubeLift(_RegisteredTask, IsaacLabEnv):
    env_id = "FrankaCubeLift-v0"
