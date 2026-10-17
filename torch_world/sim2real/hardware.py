"""Lazy LeRobot integration, with an injectable driver for motor-free tests."""

import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from torch_world.envs.robot_cube import _build
from torch_world.sim2real.config import ChessTaskConfig, SO101Config, fingerprint
from torch_world.sim2real.control import Observation, SO101Controller
from torch_world.sim2real.scene import chess_scene


class MockDriver:
    """In-memory position driver. No serial devices or cameras are opened."""

    def __init__(self, calibration, joints=(0, 0, 0, 1.57, 0), opening=1.0):
        self.calibration = calibration
        self.positions = calibration.to_driver(joints, opening)
        self.sent = []
        self.connected = False
        self.age = 0.0
        self.fail = False
        self.max_delta = None
        self.released = False

    def connect(self):
        self.connected = True

    def get_observation(self):
        if not self.connected or self.fail:
            raise ConnectionError("Mock driver disconnected")
        w, h = self.calibration.resolution
        return {
            **self.positions,
            "wrist": np.zeros((h, w, 3), dtype=np.uint8),
            "timestamp": time.perf_counter() - self.age,
        }

    def get_positions(self):
        if not self.connected or self.fail:
            raise ConnectionError("Mock driver disconnected")
        return self.positions.copy()

    def send_action(self, action):
        if not self.connected or self.fail:
            raise ConnectionError("Mock driver disconnected")
        result = {
            key: float(value)
            if self.max_delta is None
            else float(np.clip(value, self.positions[key] - self.max_delta, self.positions[key] + self.max_delta))
            for key, value in action.items()
        }
        self.positions.update(result)
        self.sent.append(result.copy())
        return result

    def disconnect(self):
        self.connected = False

    def release_torque(self):
        self.released = True


class _LeRobotDriver:
    def __init__(self, calibration, port, camera_index, config, calibration_dir):
        import sys
        from importlib.metadata import version

        if sys.version_info < (3, 12):
            raise RuntimeError("SO101 hardware requires Python 3.12+ and torch-world[so101]")
        try:
            if version("lerobot") != "0.6.1":
                raise RuntimeError("This adapter is tested with lerobot==0.6.1; install torch-world[so101]")
            from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        except ImportError as error:
            raise ImportError("Install torch-world[so101] in a Python 3.12+ environment") from error
        self.robot = SO101Follower(
            SO101FollowerConfig(
                port=port,
                id=calibration.robot_id,
                calibration_dir=None if calibration_dir is None else Path(calibration_dir),
                use_degrees=True,
                disable_torque_on_disconnect=False,
                max_relative_target={
                    **{
                        n: float(np.rad2deg(config.max_joint_step))
                        for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
                    },
                    "gripper": abs(calibration.gripper_open - calibration.gripper_closed) * config.max_gripper_step,
                },
                cameras={},
            )
        )
        self.camera = OpenCVCamera(
            OpenCVCameraConfig(
                index_or_path=camera_index, width=calibration.resolution[0], height=calibration.resolution[1], fps=30
            )
        )

    def connect(self):
        try:
            self.camera.connect()
            self.robot.connect(calibrate=False)
            if not self.robot.is_calibrated:
                raise RuntimeError("Complete LeRobot motor calibration before connecting for an experiment")
        except BaseException:
            self.disconnect()
            raise

    def get_observation(self):
        positions = self.robot.get_observation()
        # Snapshot the frame and its acquisition timestamp together, not the
        # time get_observation happens to retrieve a previously cached frame.
        with self.camera.frame_lock:
            frame, stamp = self.camera.latest_frame, self.camera.latest_timestamp
            if frame is None or stamp is None:
                raise RuntimeError("Wrist camera has no frame")
            return {**positions, "wrist": frame.copy(), "timestamp": stamp}

    def get_positions(self):
        return self.robot.get_observation()

    def send_action(self, action):
        return self.robot.send_action(action)

    def disconnect(self):
        try:
            if self.robot.is_connected:
                self.robot.disconnect()
        finally:
            if self.camera.is_connected:
                self.camera.disconnect()

    def release_torque(self):
        self.robot.bus.disable_torque()


class SO101Robot:
    """Physical execution is explicit; the default backend is an in-memory dry run.

    Construction never connects. ``reset`` clears policy/controller state from
    measurement; it does not move, home, or reset the physical chess scene.
    """

    def __init__(
        self,
        config: SO101Config,
        task: ChessTaskConfig,
        calibration,
        *,
        port=None,
        camera_index=0,
        hardware=False,
        dry_run=True,
        driver=None,
        calibration_dir=None,
    ):
        self.config, self.task, self.calibration = config, task, calibration
        self.hardware, self.dry_run = hardware, dry_run
        if hardware and (not calibration.verified or not task.measured or not port):
            raise ValueError("Hardware requires verified calibration, measured task geometry and an explicit port")
        if hardware and dry_run:
            raise ValueError("Use hardware=False for a motor-free dry run, or explicitly set dry_run=False")
        if hardware and driver is not None:
            raise ValueError("Custom drivers are test backends; do not combine with hardware=True")
        self.controller = SO101Controller(
            config, _build("so101", Path.home() / ".cache/torch_world/menagerie", chess_scene(task, "pawn", "white"))
        )
        self.driver = driver or (
            _LeRobotDriver(calibration, port, camera_index, config, calibration_dir)
            if hardware
            else MockDriver(calibration)
        )
        self.connected, self.stopped = False, False
        self.last_observation, self.last_command = None, None
        self.log = []
        self._last_send = None

    @property
    def contract(self):
        return fingerprint(self.config, self.task, self.calibration)

    def connect(self):
        try:
            self.driver.connect()
            self.connected = True
            self.reset()
            return self
        except BaseException:
            self.close()
            raise

    def observe(self):
        if not self.connected:
            raise RuntimeError("Robot is not connected")
        raw = self.driver.get_observation()
        timestamp = float(raw["timestamp"])
        age = time.perf_counter() - timestamp
        if not np.isfinite(timestamp) or age < -0.01 or age > self.config.max_observation_age:
            raise TimeoutError("Stale or invalid wrist camera timestamp")
        joints, opening = self.calibration.from_driver(raw)
        target = self.controller.pose(joints) if self.controller.target is None else self.controller.target
        commanded_opening = opening if self.controller.target is None else self.controller.opening
        obs = Observation(
            self.calibration.preprocess(raw["wrist"], self.config.size),
            torch.tensor(joints, dtype=torch.float32),
            opening,
            timestamp,
            self.contract,
            torch.tensor(np.r_[target, commanded_opening], dtype=torch.float32),
        )
        self.last_observation = obs
        return obs

    def reset(self):
        observation = self.observe()
        self.controller.reset(observation.joints.numpy(), observation.opening)
        observation.commanded = torch.tensor(
            np.r_[self.controller.target, self.controller.opening], dtype=torch.float32
        )
        self.stopped = False
        return observation

    def step(self, action, *, observation=None):
        if self.stopped:
            raise RuntimeError("Execution stopped; inspect the cause and reset from measurement")
        try:
            obs = observation or self.observe()
            age = time.perf_counter() - obs.timestamp
            if not np.isfinite(age) or age < -0.01 or age > self.config.max_observation_age:
                raise TimeoutError("Action was planned from a stale observation")
            if self._last_send is not None:
                remaining = self.config.control_dt - (time.perf_counter() - self._last_send)
                if remaining > 0:
                    time.sleep(remaining)
            if time.perf_counter() - obs.timestamp > self.config.max_observation_age:
                raise TimeoutError("Observation expired before dispatch")
            before, old_opening = self.controller.target.copy(), self.controller.opening
            command = self.controller.command(action, obs.joints.numpy(), obs.opening)
            requested_driver = self.calibration.to_driver(command.joints, command.opening)
            sent = None if self.dry_run else self.driver.send_action(requested_driver)
            self._last_send = time.perf_counter()
            if sent is not None:
                q, opening = self.calibration.from_driver(sent)
                if not np.allclose(q, command.joints, atol=1e-6) or abs(opening - command.opening) > 1e-6:
                    self.controller.reconcile(q, opening)
                    command.applied = np.r_[
                        (self.controller.target - before) / self.config.max_step, opening - old_opening
                    ]
                    command.joints, command.opening, command.target = q, opening, self.controller.target.copy()
            self.last_command = command
            self.log.append(
                {
                    "timestamp": self._last_send,
                    "observation_timestamp": obs.timestamp,
                    "requested": command.requested.tolist(),
                    "applied": command.applied.tolist(),
                    "driver_requested": requested_driver,
                    "driver_applied": sent,
                    "dry_run": self.dry_run,
                }
            )
            if self.dry_run:
                self.controller.reset(obs.joints.numpy(), obs.opening)
            time.sleep(self.config.control_dt)
            return self.observe()
        except BaseException:
            self.stop()
            raise

    def stop(self):
        """Stop dispatch and attempt a measured-position hold; never release torque."""
        self.stopped = True
        if self.connected and not self.dry_run:
            try:
                raw = self.driver.get_positions()
                joints, opening = self.calibration.from_driver(raw)
                self.driver.send_action(self.calibration.to_driver(joints, opening))
            except Exception as error:  # noqa: BLE001 -- a failed hold must not mask the original hardware fault
                warnings.warn(f"Could not confirm position hold: {error}", RuntimeWarning, stacklevel=2)

    def release_torque(self):
        """Explicit operator action, after supporting the arm and any held object."""
        self.stopped = True
        if self.connected and not self.dry_run:
            self.driver.release_torque()

    def close(self):
        if self.connected:
            self.stop()
            try:
                self.driver.disconnect()
            finally:
                self.connected = False

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()


def save_goal(path, observation, config, task, calibration):
    """Save a measured goal photo/arm pose; this is an evaluation input, not training data."""
    torch.save(
        {
            "schema": 1,
            "contract": fingerprint(config, task, calibration),
            "rgb": observation.rgb.cpu(),
            "joints": observation.joints.cpu(),
            "opening": observation.opening,
            "timestamp": observation.timestamp,
            "commanded": None if observation.commanded is None else observation.commanded.cpu(),
            "calibration": asdict(calibration),
        },
        path,
    )


def load_goal(path, config, task, calibration):
    goal = torch.load(path, map_location="cpu", weights_only=True)
    if goal.get("schema") != 1 or goal.get("contract") != fingerprint(config, task, calibration):
        raise ValueError("Goal camera/control/calibration contract does not match this experiment")
    return Observation(
        goal["rgb"], goal["joints"], goal["opening"], goal["timestamp"], goal["contract"], goal.get("commanded")
    )
