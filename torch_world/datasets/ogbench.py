from pathlib import Path

import torch
from torch import Tensor


def load_ogbench(
    name: str, split: str = "train", dataset_dir: str | Path = "~/.ogbench/data", info: bool = False
) -> dict[str, Tensor]:
    """Episodes of an OGBench dataset as tensors ``(N, T, ...)``, ready for a `TrajectoryDataset`.

    ``observations`` are float32 state vectors, or ``visual-`` uint8 images ``(N, T, 3, H, W)`` kept in uint8 to
    save memory: divide by 255 for [0, 1]. ``actions`` are ``(N, T, A)``; the action after the last frame of an
    episode has no next frame, as in OGBench. ``info=True`` adds ``qpos``, ``qvel`` and, where recorded,
    ``button_states``. Missing files are downloaded with OGBench into ``dataset_dir``.
    """
    import numpy as np

    if split not in ("train", "val"):
        raise ValueError("split must be 'train' or 'val'")
    if "-singletask" in name:
        raise ValueError("Single-task data is the goal-conditioned dataset relabelled; load it by its base name")
    dataset_dir = Path(dataset_dir).expanduser()
    path = dataset_dir / f"{name}{'-val' if split == 'val' else ''}.npz"
    if not path.is_file():
        try:
            import ogbench
        except ImportError as exc:
            raise ImportError("Install OGBench to download its datasets: pip install 'torch-world[ogbench]'") from exc
        ogbench.download_datasets([name], str(dataset_dir))
    with np.load(path) as file:
        # Each episode ends at a terminal step; the file stores episodes back to back.
        ends = np.flatnonzero(file["terminals"]) + 1
        lengths = np.unique(np.diff(ends, prepend=0))
        if len(lengths) != 1 or ends[-1] != len(file["terminals"]):
            raise ValueError(f"{name} has episodes of different lengths {lengths.tolist()}; cannot stack them")
        keys = ["observations", "actions"] + [k for k in ("qpos", "qvel", "button_states") if info and k in file]
        episodes = {k: torch.from_numpy(file[k].reshape(len(ends), int(lengths[0]), *file[k].shape[1:])) for k in keys}
    obs = episodes["observations"]
    if obs.dtype == torch.uint8:
        episodes["observations"] = obs.permute(0, 1, 4, 2, 3)
    else:
        episodes["observations"] = obs.float()
    episodes["actions"] = episodes["actions"].float()
    return episodes
