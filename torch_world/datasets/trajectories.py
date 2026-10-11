from torch import Tensor
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """Sliding windows of ``window`` steps over episodes stored as tensors ``(N, T, ...)`` with a shared ``T``."""

    def __init__(self, episodes: dict[str, Tensor], window: int):
        self.episodes = episodes
        self.window = window
        self.num_episodes, self.length = next(iter(episodes.values())).shape[:2]
        self.per_episode = self.length - window + 1

    def __len__(self) -> int:
        return self.num_episodes * self.per_episode

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        e, t = divmod(i, self.per_episode)
        return {k: v[e, t : t + self.window] for k, v in self.episodes.items()}
