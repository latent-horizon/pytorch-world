"""Render the Environments page thumbnails of the OGBench, RoboDojo and RoboCasa tasks.

OGBench: a segment of a real episode from the task's OGBench validation dataset (expert or oracle data), replayed by
restoring its recorded simulator states and rendering OGBench's own camera. These are dataset previews, not verified
successes on an evaluation task. RoboDojo: the official task video from robodojo-benchmark.com, cropped and resized.
RoboCasa: the first human demonstration of each target task, from RoboCasa's pretraining dataset, or its target
dataset for the tasks unseen in pretraining; datasets fetched here are removed afterwards.

    python examples/envs/render_benchmark_thumbnails.py [ogbench|robodojo|robocasa ...]

Needs ffmpeg, the ogbench extra for OGBench and a RoboCasa install for RoboCasa. GIFs are written to
docs/assets/envs/{ogbench,robodojo,robocasa}/.
"""

import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SIZE, FRAMES, MS = 160, 32, 200  # as the built-in success GIFs: 160 x 160 at 5 frames per second
OGBENCH_DATA = Path("~/.ogbench/data").expanduser()
ROBODOJO_VIDEO = "https://media.luminis-sim.com/media/home/sim-v2/{}.mp4"
# OGBench steps between thumbnail frames: locomotion is slower on screen than the arm.
STRIDE = {"pointmaze": 10, "antmaze": 10, "humanoidmaze": 15, "antsoccer": 10}
# The overview camera leaves the humanoid and the soccer ant a few pixels wide; follow them as the visual tasks do.
CAMERA = {"humanoidmaze-medium-navigate-v0": "back", "antsoccer-arena-navigate-v0": "back"}


def download(url, path, attempts=5):
    """Fetch ``url`` to ``path``, retrying stalled connections."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
            return
        except OSError:
            if attempt == attempts - 1:
                raise


def save_gif(frames, path, colors=64):
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(f).convert("P", palette=Image.Palette.ADAPTIVE, colors=colors) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=MS, loop=0, optimize=True)
    print(f"{path.relative_to(ROOT)}: {len(frames)} frames, {path.stat().st_size // 1024} KB", flush=True)


def ogbench_episode(name, dataset_dir):
    """Recorded states of the first validation episode of the state dataset behind ``name``."""
    from ogbench.utils import DATASET_URL

    base = name.removeprefix("visual-")
    path = dataset_dir / f"{base}-val.npz"
    if not path.is_file():
        dataset_dir.mkdir(parents=True, exist_ok=True)
        download(f"{DATASET_URL}/{base}-val.npz", path)
    with np.load(path) as file:
        end = int(np.flatnonzero(file["terminals"])[0]) + 1
        return {k: file[k][:end] for k in ("qpos", "qvel", "button_states") if k in file}


def ogbench_segment(name, states):
    """The segment of ``FRAMES`` frames in which the scene changes most, so the thumbnail shows the task."""
    family = name.removeprefix("visual-").split("-")[0]
    stride = STRIDE.get(family, 5)
    length = (FRAMES - 1) * stride
    if family in STRIDE:  # the agent's travel in the plane
        moved = states["qpos"][:, :2]
    else:  # the objects: every joint after the arm's 14, and the buttons
        moved = np.concatenate([states["qpos"][:, 14:], states.get("button_states", states["qpos"][:, :0])], 1)
    change = np.linalg.norm(moved[length:] - moved[:-length], axis=1)
    start = int(change.argmax())
    return range(start, start + length + 1, stride)


def ogbench(dataset_dir=OGBENCH_DATA):
    import ogbench as upstream

    from torch_world.envs import list_envs

    for spec in list_envs(backend="ogbench"):
        name = spec.kwargs["name"]
        states = ogbench_episode(name, dataset_dir)
        camera = {"camera_name": CAMERA[name]} if name in CAMERA else {}
        env = upstream.make_env_and_datasets(name, env_only=True, width=SIZE, height=SIZE, **camera)
        env.reset(seed=0)
        frames = []
        for t in ogbench_segment(name, states):
            extra = (states["button_states"][t],) if "button_states" in states else ()
            env.unwrapped.set_state(states["qpos"][t], states["qvel"][t], *extra)
            frames.append(env.render())
        env.close()
        save_gif(frames, ROOT / "docs/assets/envs/ogbench" / f"{name.removesuffix('-v0')}.gif")


def video_frames(video, frames):
    """A whole video as at most ``frames`` RGB frames, cropped to a centered square of ``SIZE`` pixels."""
    probe = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video]
    duration = float(subprocess.run(probe, capture_output=True, text=True, check=True).stdout)
    fps = min(1000 / MS, frames / duration)
    scale = f"fps={fps},crop=min(iw\\,ih):min(iw\\,ih),scale={SIZE}:{SIZE}:flags=area"
    decode = ["ffmpeg", "-v", "error", "-i", video, "-vf", scale, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(decode, capture_output=True, check=True).stdout
    return list(np.frombuffer(raw, np.uint8).reshape(-1, SIZE, SIZE, 3))


def robodojo():
    from torch_world.datasets.robodojo import ROBODOJO_TASKS

    with tempfile.TemporaryDirectory() as tmp:
        for task in ROBODOJO_TASKS:
            video = Path(tmp) / f"{task}.mp4"
            download(ROBODOJO_VIDEO.format(task), video)
            # At most 24 frames and 48 colors: busy real scenes compress poorly as GIFs.
            save_gif(video_frames(video, 24), ROOT / "docs/assets/envs/robodojo" / f"{task}.gif", colors=48)


def robocasa():
    from robocasa.scripts.download_datasets import download_datasets
    from robocasa.utils.dataset_registry_utils import get_ds_path

    from torch_world.envs.catalog import ROBOCASA

    for task_set, tasks in ROBOCASA.items():
        split = "target" if task_set == "composite_unseen" else "pretrain"  # unseen tasks have no pretraining data
        for task in tasks:
            path = Path(get_ds_path(task, "human", split))
            fetched = not path.exists()
            if fetched:
                download_datasets(split=[split], tasks=[task], source=["human"])
            video = path / "videos/chunk-000/observation.images.robot0_agentview_left/episode_000000.mp4"
            save_gif(video_frames(video, 24), ROOT / "docs/assets/envs/robocasa" / f"{task}.gif", colors=48)
            if fetched:
                shutil.rmtree(path)


if __name__ == "__main__":
    for suite in sys.argv[1:] or ["ogbench", "robodojo", "robocasa"]:
        {"ogbench": ogbench, "robodojo": robodojo, "robocasa": robocasa}[suite]()
