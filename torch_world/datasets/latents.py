import bisect
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


def write_latent_shard(path: str | Path, **tensors: Tensor) -> Path:
    """Save one shard of clips, ``{name: (N, ...)}`` with a shared ``N``, atomically as ``path``."""
    sizes = {v.size(0) for v in tensors.values()}
    if len(sizes) != 1:
        raise ValueError("Every tensor of a shard needs the same number of clips")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, tmp)
    tmp.replace(path)
    return path


class LatentClips(Dataset):
    """Clips precomputed into shards on disk, such as video-autoencoder latents with their actions and goals.

    Every ``*.pt`` file in ``root`` holds ``{name: (N, ...)}`` tensors written by ``write_latent_shard``, for
    example ``latents``, ``actions`` and ``goal``. Shards are memory-mapped, so a dataset larger than RAM streams from
    disk, and each worker of a ``DataLoader`` opens only the shards it reads. Item ``i`` is ``{name: tensor[i]}``.
    """

    def __init__(self, root: str | Path, pattern: str = "*.pt"):
        self.paths = sorted(Path(root).glob(pattern))
        if not self.paths:
            raise FileNotFoundError(f"No shards matching {pattern} in {root}")
        sizes = []
        for path in self.paths:
            shard = torch.load(path, mmap=True, weights_only=True)
            sizes.append(next(iter(shard.values())).size(0))
        self.offsets = [0]
        for size in sizes:
            self.offsets.append(self.offsets[-1] + size)
        self._open: dict[int, dict[str, Tensor]] = {}

    def __len__(self) -> int:
        return self.offsets[-1]

    def _shard(self, s: int) -> dict[str, Tensor]:
        if s not in self._open:
            self._open[s] = torch.load(self.paths[s], mmap=True, weights_only=True)
        return self._open[s]

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        if not 0 <= i < len(self):
            raise IndexError(i)
        s = bisect.bisect_right(self.offsets, i) - 1
        return {k: v[i - self.offsets[s]] for k, v in self._shard(s).items()}
