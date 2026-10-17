"""Camera-only inspection. Capturing a frame never connects to a motor driver."""

import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def capture_wrist_camera(path, *, device, width=1280, height=720, warmup=2.0, timeout=20.0):
    """Save one unmodified RGB frame and an adjacent JSON acquisition report.

    On macOS, ``device`` is an AVFoundation camera name or index and FFmpeg is
    required. On Linux it is an explicit V4L2 device path. There is no fallback
    to a different camera, no microphone capture, and no serial/motor connection.
    The optional notebooks dependencies provide Pillow for image inspection.
    """
    from PIL import Image

    name = str(device)
    if not name or ":" in name or width < 1 or height < 1 or not 0 <= warmup < timeout:
        raise ValueError("Specify a camera, positive dimensions, and timeout greater than warmup")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Camera-only capture requires FFmpeg on PATH")
    if sys.platform == "darwin":
        source = ["-f", "avfoundation", "-framerate", "30", "-video_size", f"{width}x{height}", "-i", f"{name}:none"]
    elif sys.platform.startswith("linux") and name.startswith("/dev/video"):
        source = ["-f", "v4l2", "-video_size", f"{width}x{height}", "-i", name]
    else:
        raise ValueError("Use a macOS AVFoundation camera name/index or a Linux /dev/video device")
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("Capture destination must be a PNG file")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="so101-camera-") as directory:
        frame_path = Path(directory) / "frame.png"
        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            *source,
            "-ss",
            str(warmup),
            "-frames:v",
            "1",
            "-update",
            "1",
            "-y",
            str(frame_path),
        ]
        try:
            subprocess.run(command, capture_output=True, text=True, check=True, timeout=timeout)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Camera capture failed for {name}: {error.stderr.strip()}") from error
        captured = datetime.now(timezone.utc).isoformat()
        with Image.open(frame_path) as image:
            rgb = np.asarray(image.convert("RGB"))
            actual_width, actual_height = image.size
        intensity = (rgb.astype(float) / 255) @ np.array([0.2126, 0.7152, 0.0722])
        report = {
            "device": name,
            "backend": source[1],
            "captured_at_utc": captured,
            "width": actual_width,
            "height": actual_height,
            "channels": "RGB",
            "mean_rgb_0_255": rgb.mean(axis=(0, 1)).round(2).tolist(),
            "mean_luma_0_1": round(float(intensity.mean()), 4),
            "dark_pixel_fraction_luma_below_0_1": round(float((intensity < 0.1).mean()), 4),
            "image": path.name,
            "motor_connection": False,
            "image_processing": "none",
        }
        shutil.copyfile(frame_path, path)
    path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report
