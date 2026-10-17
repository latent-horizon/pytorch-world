import time
from dataclasses import replace

import numpy as np
import pytest
import torch

from torch_world.sim2real import (
    Calibration,
    ChessTaskConfig,
    MockDriver,
    SO101Config,
    SO101Robot,
    fit_joint_mapping,
    load_goal,
    save_goal,
)
from torch_world.sim2real.control import SO101Controller

pytest.importorskip("mujoco")


def test_tilted_approach_reaches_off_centre_board_and_binds_contract():
    from dataclasses import asdict

    from torch_world.sim2real.config import fingerprint

    config = SO101Config()
    legacy = asdict(config)
    legacy.pop("approach_pitch")
    assert fingerprint(config) == fingerprint(legacy)
    tilted = replace(config, approach_pitch=1.2)
    assert fingerprint(config) != fingerprint(tilted)
    controller = SO101Controller(tilted)
    for target in ([0.31, -0.035, 0.096], [0.31, 0.08, 0.04]):
        joints, residual = controller.solve([0, 0, 0, 1.57, 0], target, iterations=300)
        assert residual < 0.002
        np.testing.assert_allclose(controller.pose(joints), target, atol=0.002)
    with pytest.raises(ValueError, match="approach_pitch"):
        SO101Config(approach_pitch=float("nan"))


def test_mapping_round_trip_and_fit(calibration, tmp_path):
    q = np.array([0.1, -0.2, 0.3, 1.2, -0.4])
    measured, opening = calibration.from_driver(calibration.to_driver(q, 0.37))
    np.testing.assert_allclose(measured, q)
    assert opening == pytest.approx(0.37)
    calibration.save(tmp_path / "calibration.json")
    assert Calibration.load(tmp_path / "calibration.json").id == calibration.id
    poses = np.array([q - 0.2, q, q + 0.2])
    real = np.array(
        [
            [
                calibration.to_driver(p, 1)[f"{n}.pos"]
                for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
            ]
            for p in poses
        ]
    )
    fit = fit_joint_mapping(poses, real)
    np.testing.assert_allclose(fit["signs"], calibration.signs)
    np.testing.assert_allclose(fit["offsets_degrees"], calibration.offsets_degrees)
    with pytest.raises(ValueError, match="span"):
        fit_joint_mapping(np.zeros((3, 5)), np.zeros((3, 5)))


def test_controller_limits_and_rejection():
    controller = SO101Controller(SO101Config())
    q = np.array([0.0, 0, 0, 1.57, 0])
    controller.reset(q, 1.0)
    command = controller.command([1, 0, 0, -1], q, 1.0)
    assert np.max(np.abs(command.joints - q)) <= controller.config.max_joint_step + 1e-10
    assert command.opening == pytest.approx(0.85)
    for invalid in ([float("nan")] * 4, [0, 0, 0]):
        with pytest.raises(ValueError):
            controller.command(invalid, q, 1.0)
    controller.target = np.array([0.32, 0.14, 0.18])
    with pytest.raises(ValueError, match="Unreachable"):
        controller.command([0, 0, 0, 0], q, 1.0)


def test_hardware_is_explicit_and_dry_run_never_sends(calibration):
    config, task = SO101Config(size=16, control_dt=0.002), ChessTaskConfig()
    with pytest.raises(ValueError, match="verified"):
        SO101Robot(config, task, calibration, hardware=True, dry_run=False, port="test")
    driver = MockDriver(calibration)
    with SO101Robot(config, task, calibration, driver=driver) as robot:
        robot.step([0, 0, 0, -0.1])
        assert not driver.sent
    assert not driver.connected and not driver.released


def test_mock_dispatch_actual_commands_staleness_and_cleanup(calibration, tmp_path):
    config, task = SO101Config(size=16, control_dt=0.002), ChessTaskConfig()
    driver = MockDriver(calibration)
    driver.max_delta = 0.1
    with SO101Robot(config, task, calibration, driver=driver, dry_run=False) as robot:
        obs = robot.observe()
        save_goal(tmp_path / "goal.pt", obs, config, task, calibration)
        restored = load_goal(tmp_path / "goal.pt", config, task, calibration)
        assert torch.equal(restored.rgb, obs.rgb)
        torch.testing.assert_close(restored.state(True), obs.state(True))
        legacy = torch.load(tmp_path / "goal.pt", weights_only=True)
        legacy.pop("commanded")
        torch.save(legacy, tmp_path / "legacy-goal.pt")
        old_goal = load_goal(tmp_path / "legacy-goal.pt", config, task, calibration)
        assert old_goal.state().shape == (6,)
        with pytest.raises(ValueError, match="recorded commanded"):
            old_goal.state(True)
        with pytest.raises(ValueError, match="contract"):
            load_goal(tmp_path / "goal.pt", replace(config, max_step=0.004), task, calibration)
        robot.step([0.1, 0, 0, -0.15])
        observed = robot.observe()
        np.testing.assert_allclose(observed.commanded[:3], robot.controller.target, atol=1e-7)
        assert observed.commanded[3] == pytest.approx(robot.controller.opening)
        assert robot.log[-1]["driver_applied"] == driver.sent[-1]
        assert abs(robot.last_command.applied[3]) < 0.15
        robot.last_observation.timestamp = time.perf_counter() - 5
        with pytest.raises(TimeoutError):
            robot.step([0, 0, 0, 0], observation=robot.last_observation)
        assert robot.stopped
        with pytest.raises(RuntimeError, match="stopped"):
            robot.step([0, 0, 0, 0])
        robot.reset()
        driver.age = 5
        with pytest.raises(TimeoutError):
            robot.observe()
    assert not driver.connected and not driver.released


def test_failed_driver_does_not_mask_error(calibration):
    driver = MockDriver(calibration)
    robot = SO101Robot(SO101Config(size=16), ChessTaskConfig(), calibration, driver=driver, dry_run=False)
    robot.connect()
    driver.fail = True
    with pytest.warns(RuntimeWarning, match="hold"), pytest.raises(ConnectionError):
        robot.step([0, 0, 0, 0])
    with pytest.warns(RuntimeWarning, match="hold"):
        robot.close()
    assert not driver.connected


def test_camera_frame_convention(calibration):
    controller = SO101Controller(SO101Config(size=32))
    calibration.apply_camera(controller.model, "wrist")
    camera = controller.model.camera("wrist")
    np.testing.assert_allclose(controller.model.cam_intrinsic[camera.id, :2], [1, 1])
    assert np.all(np.isfinite(camera.pos))
    image = calibration.preprocess(np.zeros((32, 32, 3), dtype=np.uint8), 16)
    assert image.shape == (3, 16, 16)
    with pytest.raises(ValueError, match="proper rigid"):
        replace(calibration, wrist_from_camera=np.zeros((4, 4)).tolist())
