import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from torch_world.sim2real import camera


def test_camera_only_capture_saves_native_frame_and_report(monkeypatch, tmp_path):
    monkeypatch.setattr(camera.sys, "platform", "darwin")
    monkeypatch.setattr(camera.shutil, "which", lambda name: "/usr/local/bin/ffmpeg")

    def capture(command, **kwargs):
        assert command[command.index("-i") + 1] == "icspring camera:none"
        assert command[command.index("-frames:v") + 1] == "1"
        assert kwargs["check"] and kwargs["timeout"] == 20
        Image.new("RGB", (1280, 720), (32, 128, 224)).save(command[-1])

    monkeypatch.setattr(camera.subprocess, "run", capture)
    path = tmp_path / "camera.png"
    report = camera.capture_wrist_camera(path, device="icspring camera")
    assert (report["width"], report["height"]) == (1280, 720)
    assert report["mean_rgb_0_255"] == [32, 128, 224]
    assert report["motor_connection"] is False
    assert report["image_processing"] == "none"
    assert json.loads(path.with_suffix(".json").read_text()) == report
    with Image.open(path) as image:
        assert image.getpixel((0, 0)) == (32, 128, 224)


def test_failed_capture_does_not_report_a_stale_frame(monkeypatch, tmp_path):
    monkeypatch.setattr(camera.sys, "platform", "linux")
    monkeypatch.setattr(camera.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    path = tmp_path / "camera.png"
    path.write_bytes(b"existing capture")

    def fail(command, **kwargs):
        assert command[command.index("-i") + 1] == "/dev/video2"
        raise subprocess.CalledProcessError(1, command, stderr="device unavailable")

    monkeypatch.setattr(camera.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="device unavailable"):
        camera.capture_wrist_camera(path, device="/dev/video2")
    assert path.read_bytes() == b"existing capture"
    assert not Path(path.with_suffix(".json")).exists()
