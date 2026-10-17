import pytest

from torch_world.sim2real import Calibration


@pytest.fixture
def calibration():
    return Calibration(
        robot_id="mock-so101",
        signs=(1, -1, 1, -1, 1),
        offsets_degrees=(0, 5, -5, 10, 0),
        gripper_closed=0,
        gripper_open=100,
        intrinsics=(32, 32, 16, 16),
        resolution=(32, 32),
        wrist_from_camera=((1, 0, 0, -0.06), (0, 1, 0, 0), (0, 0, 1, 0.035), (0, 0, 0, 1)),
    )
