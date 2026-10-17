"""Library-owned torque-controlled arm reaching around a fixed obstacle."""

from torch_world.envs.physics_tasks import MujocoTask


class ObstacleArm(MujocoTask):
    """Two-joint MuJoCo arm with RGB observations and pose-and-velocity success.

    Actions are normalized shoulder/elbow torques. States contain their joint
    positions followed by velocities. Rendering a target never advances physics.
    Install ``torch-world[lewam]`` for the recorded simulator version.
    """

    env_id = "ObstacleArm-v0"

    def __init__(self, **kwargs):
        super().__init__("ObstacleArm", **kwargs)
