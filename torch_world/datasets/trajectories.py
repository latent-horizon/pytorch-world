from torch import Tensor
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """Sliding windows of ``window`` steps over episodes stored as tensors ``(N, T, ...)`` with a shared ``T``.

    With a ``frameskip`` of ``k``, a window keeps every ``k``-th frame and each of its steps concatenates the ``k``
    environment actions that follow the frame, so ``actions`` ``(N, T, A)`` become ``(window, k * A)``, as DINO-WM and
    LeWM train. ``stacked`` names the keys whose ``k`` steps are concatenated; the others are subsampled.
    """

    def __init__(
        self, episodes: dict[str, Tensor], window: int, frameskip: int = 1, stacked: tuple[str, ...] = ("actions",)
    ):
        self.episodes = episodes
        self.window, self.frameskip, self.stacked = window, frameskip, stacked
        self.num_episodes, self.length = next(iter(episodes.values())).shape[:2]
        self.per_episode = self.length - window * frameskip + 1

    def __len__(self) -> int:
        return self.num_episodes * self.per_episode

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        e, t = divmod(i, self.per_episode)
        k, end = self.frameskip, t + self.window * self.frameskip
        return {
            key: v[e, t:end].reshape(self.window, k * v.size(-1)) if key in self.stacked else v[e, t:end:k]
            for key, v in self.episodes.items()
        }
