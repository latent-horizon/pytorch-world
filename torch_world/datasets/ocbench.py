from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor

from torch_world.datasets.images import resize_images

OCBENCH_CAMERAS = ("front", "side", "ur5e/wrist")


def _resolve_camera(camera: str | int | None) -> int | None:
    if camera is None:
        return None
    if isinstance(camera, str):
        if camera not in OCBENCH_CAMERAS:
            raise ValueError(f"Unknown OCBench camera {camera!r}; expected one of {OCBENCH_CAMERAS}")
        return OCBENCH_CAMERAS.index(camera)
    if isinstance(camera, int) and not isinstance(camera, bool):
        if camera < 0:
            raise ValueError("camera index must be non-negative")
        return camera
    raise ValueError("camera must be a camera name, an integer index, or None")


def _resolve_paths(
    name: str | Path | Sequence[str | Path],
    split: str,
    dataset_root: str | Path | None,
    num_shards: int | None,
) -> list[Path]:
    if isinstance(name, (str, Path)):
        candidate = Path(name).expanduser()
        if isinstance(name, Path) or str(name).endswith(".npz") or candidate.is_file():
            return [candidate]
        try:
            import ocbench
        except ImportError as exc:
            raise ImportError("Install OCBench to download its datasets: pip install 'torch-world[ocbench]'") from exc
        train_paths, val_paths = ocbench.download_datasets(name, dataset_root=dataset_root, num_shards=num_shards)
        selected = train_paths if split == "train" else val_paths
        return [Path(path) for path in selected]
    paths = [Path(path).expanduser() for path in name]
    if not paths:
        raise ValueError("Provide at least one dataset path")
    return paths


def load_ocbench(
    name: str | Path | Sequence[str | Path],
    split: str = "train",
    *,
    dataset_root: str | Path | None = None,
    num_shards: int | None = None,
    camera: str | int | None = "front",
    size: int | None = None,
    success_only: bool = False,
    max_episodes: int | None = None,
) -> list[dict[str, Tensor]]:
    import numpy as np

    if split not in ("train", "val"):
        raise ValueError("split must be 'train' or 'val'")
    if size is not None and size < 1:
        raise ValueError("size must be positive")
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("max_episodes must be positive")
    if num_shards is not None and num_shards < 1:
        raise ValueError("num_shards must be positive")
    camera_idx = _resolve_camera(camera)
    paths = _resolve_paths(name, split, dataset_root, num_shards)
    episodes: list[dict[str, Tensor]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as file:
            interval = int(file["observation_interval"]) if "observation_interval" in file else 1
            if interval < 1:
                raise ValueError(f"Invalid observation_interval {interval} in {path}")
            terminals = np.asarray(file["terminals"], dtype=bool)
            actions = file["actions"]
            rewards = file["rewards"]
            masks = file["masks"]
            observations = file["observations"]
            if not (len(actions) == len(rewards) == len(masks) == len(terminals)):
                raise ValueError(f"Mismatched transition array lengths in {path}")
            ends = np.flatnonzero(terminals) + 1
            if len(ends) == 0 or ends[-1] != len(terminals):
                raise ValueError(f"Invalid episode boundaries in {path}")
            starts = np.concatenate(([0], ends[:-1]))
            obs_counts = (ends - starts + interval - 1) // interval + 1
            obs_ends = np.cumsum(obs_counts)
            obs_starts = np.concatenate(([0], obs_ends[:-1]))
            if obs_ends[-1] != len(observations):
                raise ValueError(f"Invalid observation boundaries in {path}")
            for start, end, obs_start, obs_end in zip(starts, ends, obs_starts, obs_ends, strict=True):
                ep_rewards = np.asarray(rewards[start:end], dtype=np.float32)
                if success_only and not np.any(ep_rewards == 1):
                    continue
                ep_obs = observations[obs_start:obs_end]
                if ep_obs.dtype == np.uint8:
                    if ep_obs.ndim == 5:
                        if camera_idx is not None:
                            if camera_idx >= ep_obs.shape[1]:
                                raise ValueError(f"camera index {camera_idx} out of range for {ep_obs.shape[1]} views")
                            ep_obs = ep_obs[:, camera_idx]
                    elif ep_obs.ndim == 4 and camera_idx not in (None, 0):
                        raise ValueError("Single-camera dataset has only one view")
                    if size is not None:
                        obs_tensor = resize_images(ep_obs, size)
                    else:
                        obs_tensor = torch.from_numpy(np.ascontiguousarray(ep_obs)).movedim(-1, -3).contiguous()
                else:
                    obs_tensor = torch.from_numpy(np.asarray(ep_obs, dtype=np.float32))
                ep_actions = np.asarray(actions[start:end], dtype=np.float32)
                ep_masks = np.asarray(masks[start:end], dtype=np.float32)
                ep_terminals = terminals[start:end]
                chunks = int(obs_end - obs_start) - 1
                action_dim = ep_actions.shape[-1]
                if interval == 1:
                    chunk_actions = ep_actions
                    chunk_rewards = ep_rewards
                    chunk_terminated = ep_masks == 0
                    chunk_truncated = ep_terminals & (ep_masks != 0)
                    chunk_width = action_dim
                else:
                    pad = chunks * interval - int(end - start)
                    if pad:
                        padded_actions = np.concatenate(
                            [ep_actions, np.zeros((pad, action_dim), dtype=np.float32)], axis=0
                        )
                        padded_rewards = np.concatenate([ep_rewards, np.zeros((pad,), dtype=np.float32)], axis=0)
                        padded_masks = np.concatenate([ep_masks, np.ones((pad,), dtype=np.float32)], axis=0)
                    else:
                        padded_actions, padded_rewards, padded_masks = ep_actions, ep_rewards, ep_masks
                    chunk_width = interval * action_dim
                    chunk_actions = padded_actions.reshape(chunks, chunk_width)
                    chunk_rewards = padded_rewards.reshape(chunks, interval).sum(axis=1)
                    chunk_terminated = (padded_masks == 0).reshape(chunks, interval).any(axis=1)
                    chunk_truncated = np.zeros(chunks, dtype=bool)
                    chunk_truncated[-1] = bool(ep_terminals[-1]) and not bool(chunk_terminated[-1])
                episodes.append(
                    {
                        "observations": obs_tensor.unsqueeze(0),
                        "actions": torch.from_numpy(
                            np.concatenate([chunk_actions, np.zeros((1, chunk_width), dtype=np.float32)], axis=0)
                        ).unsqueeze(0),
                        "rewards": torch.from_numpy(
                            np.concatenate([chunk_rewards, np.zeros((1,), dtype=np.float32)], axis=0)
                        ).unsqueeze(0),
                        "terminated": torch.from_numpy(
                            np.concatenate([chunk_terminated, np.zeros((1,), dtype=bool)], axis=0)
                        ).unsqueeze(0),
                        "truncated": torch.from_numpy(
                            np.concatenate([chunk_truncated, np.zeros((1,), dtype=bool)], axis=0)
                        ).unsqueeze(0),
                    }
                )
                if max_episodes is not None and len(episodes) >= max_episodes:
                    return episodes
    return episodes
