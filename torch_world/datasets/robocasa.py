"""RoboCasa demonstrations in the LeRobot format RoboCasa releases, with actions in its environment's order."""

import json
from pathlib import Path

import torch
from torch import Tensor

from torch_world.datasets.images import resize_images

ACTION_LAYOUT = {
    "end_effector_position": (0, 3),
    "end_effector_rotation": (3, 6),
    "gripper_close": (6, 7),
    "base_motion": (7, 11),
    "control_mode": (11, 12),
}
"""Where each action field sits in RoboCasa's environment actions, as robocasa's ``ACTION_KEY_ORDERING_HDF5``."""


def load_robocasa(
    root: str | Path, episode: int, *, camera: str = "robot0_agentview_left", size: int = 128
) -> dict[str, Tensor]:
    """One episode of a RoboCasa dataset, a LeRobot directory, as tensors ``(1, T, ...)`` for a `TrajectoryDataset`.

    ``observations`` are ``camera`` frames ``(1, T, 3, size, size)`` in uint8. ``actions`` ``(1, T, 12)`` are reordered
    from the dataset's layout to the order `RoboCasa` environments take; ``states`` ``(1, T, 16)`` are the dataset's
    ``observation.state``: base position and rotation, end effector pose relative to the base, gripper joints.
    Episodes differ in length, so combine them with ``ConcatDataset``. Needs the ``robocasa`` extra.
    """
    import numpy as np

    try:
        import av
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Reading RoboCasa datasets needs: pip install 'torch-world[robocasa]'") from exc
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    modality = json.loads((root / "meta" / "modality.json").read_text())
    chunk = episode // info["chunks_size"]
    table = pq.read_table(root / info["data_path"].format(episode_chunk=chunk, episode_index=episode))
    stored = np.stack(table.column("action").to_numpy(zero_copy_only=False)).astype(np.float32)
    actions = np.zeros_like(stored)
    for key, (start, end) in ACTION_LAYOUT.items():
        actions[:, start:end] = stored[:, modality["action"][key]["start"] : modality["action"][key]["end"]]
    states = np.stack(table.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32)
    video = root / info["video_path"].format(
        episode_chunk=chunk, video_key=f"observation.images.{camera}", episode_index=episode
    )
    with av.open(str(video)) as container:
        frames = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
    if len(frames) != len(actions):
        raise ValueError(f"episode {episode}: {len(frames)} frames but {len(actions)} actions")
    return {
        "observations": resize_images(torch.from_numpy(frames), size)[None],
        "states": torch.from_numpy(states)[None],
        "actions": torch.from_numpy(actions)[None],
    }


def load_robocasa_scene(root: str | Path, episode: int) -> dict:
    """The kitchen and initial simulator state of one demonstration, for ``RoboCasa.reset(scenes=...)``.

    Returns ``ep_meta`` (layout, style, objects and the language instruction), ``model`` (the scene's MJCF) and
    ``state``, the flattened MuJoCo state at the first step, as RoboCasa stores them in ``extras/``.
    """
    import gzip

    import numpy as np

    extras = Path(root) / "extras" / f"episode_{episode:06d}"
    with np.load(extras / "states.npz") as states:
        state = states["states"][0]
    return {
        "ep_meta": json.loads((extras / "ep_meta.json").read_text()),
        "model": gzip.decompress((extras / "model.xml.gz").read_bytes()).decode(),
        "state": state,
    }
