import torch

from torch_world.benchmarks.goal_suite import TASKS, TemporalGoals, Windows, run_episodes
from torch_world.datasets import TrajectoryDataset


def episodes(N=6, T=40, A=2):
    g = torch.Generator().manual_seed(0)
    return {
        "pixels": torch.randint(0, 256, (N, T, 3, 8, 8), dtype=torch.uint8, generator=g),
        "actions": torch.randn(N, T, A, generator=g),
    }


def test_ten_tasks():
    assert len(TASKS) == 10 and len({t.name for t in TASKS.values()}) == 10


def test_windows_match_trajectory_dataset():
    data = episodes()
    for length, k in [(4, 5), (17, 1), (3, 2)]:
        windows = Windows(data, length, k, updates=2, batch_size=16, generator=torch.Generator().manual_seed(1))
        reference = TrajectoryDataset({"pixels": data["pixels"].float() / 255, "actions": data["actions"]}, length, k)
        batch = windows.sample()
        assert batch["pixels"].shape == (16, length, 3, 8, 8) and batch["actions"].shape == (16, length, k * 2)
        # every sampled window is one of the dataset's windows
        flat = torch.stack([reference[i]["actions"] for i in range(len(reference))]).flatten(1)
        for actions in batch["actions"].flatten(1):
            assert torch.isclose(flat, actions).all(1).any()
        assert len(list(windows)) == 2


def test_temporal_goals_reward_and_termination():
    data = episodes(T=60)
    obs, actions, rewards, terminated = TemporalGoals(data, 2, 0.5, updates=1, batch_size=256).sample()
    assert obs.shape == (256, 4, 6, 8, 8) and actions.shape == (256, 3, 4)
    # the goal is the same image at every step of the window, in the second half of the channels
    assert torch.equal(obs[:, :1, 3:].expand(-1, 4, -1, -1, -1), obs[:, :, 3:])
    # -1 per step until the goal; the goal frame, never inside the window, can only be its last step
    assert torch.equal(rewards, -(1 - terminated)) and terminated[:, :2].eq(0).all()
    assert 0 < terminated[:, 2].mean() < 1


class Counter:
    """Two lanes; lane 0 reaches its goal after three steps, lane 1 never; lane 1 is not a task."""

    steps = 5
    valid = torch.tensor([True, False])

    def reset(self):
        self.t = 0
        return torch.zeros(2, 3, 4, 4)

    def step(self, action):
        self.t += 1
        return torch.zeros(2, 3, 4, 4)

    def success(self):
        return torch.tensor([self.t == 3, False])


def test_run_episodes_counts_reaching_at_any_step():
    out = run_episodes(Counter(), lambda obs: torch.zeros(len(obs), 2), frameskip=1)
    assert out["success"] == [True] and out["first_success"] == [3] and out["ms_per_step"] >= 0
