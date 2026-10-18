"""OGBench adapter and dataset loader: protocol tests on a fake simulator, then the real package when installed."""

import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from torch_world.datasets import TrajectoryDataset, load_ogbench

HAS_OGBENCH = importlib.util.find_spec("ogbench") is not None


class FakeEnv:
    """A 1-D point that reaches its goal at 1; resets draw the goal from NumPy's global generator, as OGBench does."""

    def __init__(self, name, low=-2.0, high=2.0):
        self.name = name
        self.action_space = SimpleNamespace(low=np.full(1, low), high=np.full(1, high), shape=(1,))
        self.observation_space = SimpleNamespace(shape=(1,))
        self.spec = SimpleNamespace(max_episode_steps=3)
        self.actions, self.closed = [], False

    def reset(self, seed=None, options=None):
        self.options, self.x = options, np.zeros(1)
        return self.x.copy(), {"goal": np.random.uniform(1, 2, 1), "goal_rendered": np.zeros((4, 4, 3), np.uint8)}

    def step(self, action):
        self.actions.append(action.copy())
        self.x = self.x + action
        success = bool(self.x[0] >= 1)
        return self.x.copy(), float(success), success, False, {"success": float(success)}

    def render(self):
        return np.full((4, 4, 3), 7, np.uint8)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_ogbench(monkeypatch):
    made = []

    def make_env_and_datasets(name, env_only, **kwargs):
        assert env_only
        made.append(FakeEnv(name))
        made[-1].kwargs = kwargs
        return made[-1]

    monkeypatch.setitem(sys.modules, "ogbench", SimpleNamespace(make_env_and_datasets=make_env_and_datasets))
    return made


def test_lanes_absorb_after_success_and_actions_are_rescaled(fake_ogbench):
    from torch_world.envs import OGBench

    env = OGBench("fake-navigate-v0", batch_size=2, task_id=3)
    env.reset(seed=0)
    assert fake_ogbench[0].options == {"render_goal": True, "task_id": 3}
    env.step(torch.tensor([[0.5], [0.0]]))  # [-1, 1] maps to the upstream bounds [-2, 2]
    assert env.state[:, 0].tolist() == [1.0, 0.0]
    assert env.transition.terminated.tolist() == [True, False] and env.transition.success.tolist() == [True, False]
    env.step(torch.tensor([[0.5], [0.0]]))
    assert len(fake_ogbench[0].actions) == 1  # a finished lane is not stepped again
    assert env.transition.reward.tolist() == [0, 0] and env.transition.success.tolist() == [True, False]
    env.step(torch.zeros(2, 1))
    assert env.transition.truncated.tolist() == [False, True] and env.transition.done.all()
    env.close()
    assert all(e.closed for e in fake_ogbench)


def test_seeded_resets_own_their_global_numpy_stream(fake_ogbench):
    from torch_world.envs import OGBench

    env = OGBench("fake-navigate-v0", batch_size=2)
    np.random.seed(123)
    outer = np.random.get_state()[1].copy()
    env.reset(seed=4)
    first = env.goal.clone()
    np.random.uniform()  # the caller's draws do not shift the lanes' streams
    env.reset(seed=4)
    assert torch.equal(first, env.goal) and first[0] != first[1]
    np.random.seed(123)
    assert np.array_equal(outer, np.random.get_state()[1])
    env.reset(seed=5)
    assert not torch.equal(first, env.goal)


def test_goals_are_observations_and_single_task_names_have_none(fake_ogbench):
    from torch_world.envs import OGBench, get_spec

    env = OGBench("fake-navigate-v0", batch_size=2, render_size=16)
    obs = env.reset(seed=0)
    assert fake_ogbench[0].kwargs == {"width": 16, "height": 16}  # state tasks render presentation frames only
    assert torch.equal(env.render(env.goal), env.goal) and torch.equal(env.render_target(), env.goal)
    assert env.render_frame().dtype == torch.uint8 and env.render_goal_frame().shape == (2, 3, 4, 4)
    assert obs.shape == (2, 1) and get_spec(env).id == "OGBench/fake-navigate-v0"
    with pytest.raises(ValueError, match="checkpoints"):
        env.reset(torch.zeros(2, 1))

    single = OGBench("fake-singletask-task2-v0")
    single.reset(seed=0)
    assert "task_id" not in fake_ogbench[-1].options and not hasattr(single, "goal")
    with pytest.raises(ValueError, match="fix the task"):
        OGBench("fake-singletask-v0", task_id=2)
    with pytest.raises(ValueError, match="task_id"):
        OGBench("fake-navigate-v0", task_id=6)


def test_discrete_actions_are_rejected(fake_ogbench, monkeypatch):
    from torch_world.envs import OGBench

    discrete = FakeEnv("powderworld")
    discrete.action_space = SimpleNamespace(n=4)
    monkeypatch.setitem(sys.modules, "ogbench", SimpleNamespace(make_env_and_datasets=lambda *a, **k: discrete))
    with pytest.raises(ValueError, match="discrete"):
        OGBench("powderworld-easy-play-v0")


def test_load_ogbench_stacks_episodes(tmp_path):
    terminals = np.tile([0, 0, 1], 2)
    np.savez(
        tmp_path / "visual-fake-play-v0.npz",
        observations=np.arange(6 * 4 * 4 * 3, dtype=np.uint8).reshape(6, 4, 4, 3),
        actions=np.arange(12, dtype=np.float32).reshape(6, 2),
        terminals=terminals,
        qpos=np.zeros((6, 5)),
    )
    episodes = load_ogbench("visual-fake-play-v0", dataset_dir=tmp_path, info=True)
    assert episodes["observations"].shape == (2, 3, 3, 4, 4) and episodes["observations"].dtype == torch.uint8
    assert episodes["actions"][1, 0].tolist() == [6, 7] and episodes["qpos"].shape == (2, 3, 5)
    assert len(TrajectoryDataset(episodes, window=2)) == 4

    np.savez(
        tmp_path / "fake-val.npz", observations=np.zeros((5, 2)), actions=np.zeros((5, 1)), terminals=[0, 1, 0, 0, 1]
    )
    with pytest.raises(ValueError, match="different lengths"):
        load_ogbench("fake", split="val", dataset_dir=tmp_path)
    with pytest.raises(ValueError, match="base name"):
        load_ogbench("cube-single-play-singletask-v0", dataset_dir=tmp_path)


@pytest.mark.skipif(not HAS_OGBENCH, reason="requires the ogbench extra")
@pytest.mark.parametrize(
    ("name", "shape"),
    [
        ("pointmaze-medium-navigate-v0", (2,)),
        ("cube-single-play-v0", (28,)),
        ("visual-cube-single-play-v0", (3, 64, 64)),
    ],
)
def test_real_tasks(name, shape):
    from torch_world.envs import OGBench

    with OGBench(name, batch_size=2, render_size=32) as env:
        first = env.reset(seed=3)
        assert first.shape == (2, *shape) and env.goal.shape == (2, *shape)
        assert torch.equal(first, env.reset(seed=3)) and not torch.equal(first[0], first[1])
        obs = env.step(torch.zeros(2, env.action_dim))
        assert torch.isfinite(obs).all() and env.transition.success.dtype == torch.bool
        assert env.render_frame().dtype == torch.uint8 and env.render_goal_frame().shape[:2] == (2, 3)


def test_catalog_lists_ogbench_tasks_without_named_classes(fake_ogbench):
    from torch_world import envs
    from torch_world.envs import get_spec, list_envs, make

    specs = list_envs(backend="ogbench")
    assert len(specs) == 13 and {s.category for s in specs} == {"navigation", "manipulation"}
    assert all(s.id == f"OGBench/{s.kwargs['name']}" for s in specs)
    assert not any("/" in name for name in envs.__all__)  # `from torch_world.envs import *` stays valid
    env = make("OGBench/cube-single-play-v0", batch_size=2)
    assert get_spec(env) is get_spec("OGBench/cube-single-play-v0") and fake_ogbench[0].name == "cube-single-play-v0"
