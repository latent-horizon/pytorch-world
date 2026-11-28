import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HAS_OCBENCH = importlib.util.find_spec("ocbench") is not None


class FakeCpuEnv:
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self.unwrapped = self
        self.action_space = SimpleNamespace(low=np.full(1, -2.0), high=np.full(1, 2.0), shape=(1,))
        self.actions = []
        self.cameras = []
        self.closed = False
        self.steps = 0
        self._max_episode_steps = kwargs.get("max_episode_steps", 3)
        self._terminate_at_success = kwargs.get("terminate_at_success", True)
        self._is_bowling = name.removeprefix("visual-").startswith("bowling-")

    def reset(self, seed=None, options=None):
        self.seed = seed
        self.steps = 0
        self.x = np.array([float((seed or 0) % 7) / 10], dtype=np.float32)
        return self.x.copy(), {"success": False, "failure": False, "healthy": True}

    def step(self, action):
        self.actions.append(action.copy())
        self.steps += 1
        self.x = self.x + action
        healthy = bool(self.x[0] > -1.5)
        if self._is_bowling:
            reward = 0.1 if action[0] > 0 else 0.0
            terminated = (not healthy) or self.steps >= self._max_episode_steps
            info = {"success": False, "failure": False, "healthy": healthy}
            return self.x.copy(), reward, terminated, False, info
        success = healthy and bool(self.x[0] >= 1.0)
        terminated = (not healthy) or (self._terminate_at_success and success)
        info = {"success": success, "failure": False, "healthy": healthy}
        return self.x.copy(), float(success), terminated, False, info

    def render(self, camera=None):
        self.cameras.append(camera)
        size = self.kwargs.get("width", 4)
        fill = 128 if camera == "front" else 64
        return np.full((size, size, 3), fill, dtype=np.uint8)

    def close(self):
        self.closed = True


class FakeMjWarpEnv:
    def __init__(self, name, nworld, **kwargs):
        self.name = name
        self.nworld = nworld
        self.kwargs = kwargs
        self.single_action_space = SimpleNamespace(low=np.full(1, -2.0), high=np.full(1, 2.0), shape=(1,))
        self.actions = []
        self.cameras = []
        self.closed = False
        self.steps = 0
        self._terminate_at_success = True
        self._is_bowling = name.removeprefix("visual-").startswith("bowling-")
        self.cpu_env = SimpleNamespace(_max_episode_steps=3)

    def reset(self, seed=None, options=None, seeds=None):
        self.seeds = list(seeds)
        self.steps = 0
        self.x = np.array([[float(s % 7) / 10] for s in seeds], dtype=np.float32)
        return self.x.copy(), {
            "success": np.zeros(self.nworld, dtype=bool),
            "failure": np.zeros(self.nworld, dtype=bool),
            "healthy": np.ones(self.nworld, dtype=bool),
        }

    def step(self, action):
        self.actions.append(action.copy())
        self.steps += 1
        self.x = self.x + action
        healthy = self.x[:, 0] > -1.5
        if self._is_bowling:
            reward = np.where(action[:, 0] > 0, 0.1, 0.0).astype(np.float32)
            terminated = (~healthy) | (self.steps >= self.cpu_env._max_episode_steps)
            info = {
                "success": np.zeros(self.nworld, dtype=bool),
                "failure": np.zeros(self.nworld, dtype=bool),
                "healthy": healthy,
            }
            return self.x.copy(), reward, terminated, np.zeros(self.nworld, dtype=bool), info
        success = healthy & (self.x[:, 0] >= 1.0)
        terminated = (~healthy) | (self._terminate_at_success & success)
        info = {"success": success, "failure": np.zeros(self.nworld, dtype=bool), "healthy": healthy}
        return self.x.copy(), success.astype(np.float32), terminated, np.zeros(self.nworld, dtype=bool), info

    def render_world(self, world_id, camera=None):
        self.cameras.append((world_id, camera))
        size = self.kwargs.get("width", 4)
        fill = 128 if camera == "front" else 64
        return np.full((size, size, 3), fill, dtype=np.uint8)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_ocbench(monkeypatch):
    made = []
    valid = {
        "block-cpu-single-task1-v0": ("block", 3),
        "block-cpu-lite-single-task1-v0": ("block", 2),
        "bowling-cpu-task1-v0": ("bowling", 3),
        "chamber-cpu-easy-task1-v0": ("chamber", 4),
    }

    def parse_env_spec(name):
        base = name.removeprefix("visual-")
        family, rest = base.split("-", 1)
        backend = "cpu" if rest.startswith("cpu-") else "mjwarp"
        cpu_base = base if backend == "cpu" else f"{family}-cpu-{rest}"
        if cpu_base not in valid:
            raise KeyError(cpu_base)
        parsed_family, steps = valid[cpu_base]
        return parsed_family, backend, {}, steps

    def make(name, **kwargs):
        _, backend, _, _ = parse_env_spec(name)
        if backend == "cpu":
            env = FakeCpuEnv(name, **kwargs)
        else:
            nworld = kwargs.pop("nworld", 1024)
            env = FakeMjWarpEnv(name, nworld=nworld, **kwargs)
        made.append(env)
        return env

    monkeypatch.setitem(sys.modules, "ocbench", SimpleNamespace(parse_env_spec=parse_env_spec, make=make))
    return made


def _patch_backend(monkeypatch, gpu):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: gpu)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args, **kwargs: (
            object()
            if name == "mujoco_warp" and gpu
            else (None if name == "mujoco_warp" else real_find_spec(name, *args, **kwargs))
        ),
    )


@pytest.mark.parametrize("gpu", [False, True])
def test_lanes_absorb_after_success_on_either_backend(fake_ocbench, monkeypatch, gpu):
    from torch_world.envs import OCBench

    _patch_backend(monkeypatch, gpu)
    env = OCBench("block-single-task1-v0", batch_size=2, render_size=6)
    assert env.sim_backend == ("mjwarp" if gpu else "cpu")
    assert len(fake_ocbench) == (1 if gpu else 2)
    env.reset(seed=4)
    if gpu:
        assert fake_ocbench[0].seeds == [4, 5]
        assert fake_ocbench[0].name == "block-single-task1-v0"
    else:
        assert [e.seed for e in fake_ocbench] == [4, 5]
        assert [e.name for e in fake_ocbench] == ["block-cpu-single-task1-v0"] * 2
    env.step(torch.tensor([[0.5], [0.0]]))
    assert env.state[:, 0].tolist() == pytest.approx([1.4, 0.5])
    assert env.transition.terminated.tolist() == [True, False] and env.transition.success.tolist() == [True, False]
    assert env.transition.reward.tolist() == [1.0, 0.0]
    env.step(torch.tensor([[0.5], [0.0]]))
    assert env.state[0, 0].item() == pytest.approx(1.4)
    assert env.transition.reward.tolist() == [0.0, 0.0] and env.transition.success.tolist() == [True, False]
    if gpu:
        assert fake_ocbench[0].actions[1].tolist() == [[0.0], [0.0]]
    else:
        assert len(fake_ocbench[0].actions) == 1
    env.step(torch.zeros(2, 1))
    assert env.transition.truncated.tolist() == [False, True] and env.transition.done.all()
    assert env.render_frame().shape == (2, 3, 6, 6) and env.render_goal_frame().dtype == torch.uint8
    assert env.gt_label == "Task at reset"
    env.close()
    assert all(e.closed for e in fake_ocbench)


def test_visual_camera_options_and_validation(fake_ocbench, monkeypatch):
    from torch_world.envs import OCBench, get_spec, make

    _patch_backend(monkeypatch, False)
    env = OCBench("visual-block-single-task1-v0", batch_size=2, size=4, camera="side")
    obs = env.reset(seed=1)
    assert env.observation == "rgb" and obs.shape == (2, 3, 4, 4)
    assert obs.max().item() == pytest.approx(64 / 255) and env.state.shape == (2, 1)
    assert fake_ocbench[0].name == "visual-block-cpu-single-task1-v0"
    assert fake_ocbench[0].kwargs["ob_type"] == "states" and fake_ocbench[0].kwargs["visualize_info"] is False
    assert fake_ocbench[0].kwargs["width"] == 4 and fake_ocbench[0].kwargs["height"] == 4
    with pytest.raises(ValueError, match="checkpoints"):
        env.reset(torch.zeros(2, 1))
    with pytest.raises(ValueError, match="not both"):
        env.reset(seed=0, generator=torch.Generator())
    with pytest.raises(ValueError, match="Arbitrary-state"):
        env.render(env.state)
    with pytest.raises(ValueError, match="Arbitrary-state"):
        env.render_frame(env.state)
    env.close()

    patient = make("OCBench/block-single-task1-v0", batch_size=2, terminate_on_success=False)
    assert get_spec(patient).id == "OCBench/block-single-task1-v0"
    patient.reset(seed=4)
    patient.step(torch.tensor([[0.5], [0.0]]))
    assert patient.transition.success.tolist() == [True, False] and not patient.transition.terminated.any()
    patient.step(torch.tensor([[0.0], [-1.0]]))
    assert patient.transition.terminated.tolist() == [False, True]
    patient.close()

    _patch_backend(monkeypatch, True)
    cpu_forced = OCBench("block-cpu-single-task1-v0", batch_size=2)
    assert cpu_forced.sim_backend == "cpu"
    cpu_forced.close()
    cpu_explicit = OCBench("block-single-task1-v0", batch_size=2, sim_backend="cpu")
    assert cpu_explicit.sim_backend == "cpu"
    cpu_explicit.close()
    mjwarp_override = OCBench("block-single-task1-v0", batch_size=2, max_episode_steps=2, terminate_on_success=False)
    assert mjwarp_override.sim_backend == "mjwarp" and "max_episode_steps" not in fake_ocbench[-1].kwargs
    assert "terminate_at_success" not in fake_ocbench[-1].kwargs
    mjwarp_override.reset(seed=4)
    mjwarp_override.step(torch.tensor([[0.5], [0.0]]))
    assert (
        mjwarp_override.transition.success.tolist() == [True, False] and not mjwarp_override.transition.terminated.any()
    )
    mjwarp_override.step(torch.zeros(2, 1))
    assert mjwarp_override.transition.truncated.all()
    mjwarp_override.close()

    with pytest.raises(ValueError, match="observation"):
        OCBench("block-single-task1-v0", observation="depth")
    with pytest.raises(ValueError, match="visual-"):
        OCBench("visual-block-single-task1-v0", observation="state")
    with pytest.raises(ValueError, match="sim_backend"):
        OCBench("block-single-task1-v0", sim_backend="gpu")
    with pytest.raises(ValueError, match="-cpu-"):
        OCBench("block-cpu-single-task1-v0", sim_backend="mjwarp")
    with pytest.raises(ValueError, match="Unknown OCBench"):
        OCBench("unknown-task1-v0")


@pytest.mark.parametrize("gpu", [False, True])
def test_bowling_has_no_binary_success_and_truncates_at_limit(fake_ocbench, monkeypatch, gpu):
    from torch_world.envs import OCBench

    _patch_backend(monkeypatch, gpu)
    with OCBench("bowling-task1-v0", batch_size=2, max_episode_steps=2) as env:
        env.reset(seed=0)
        env.step(torch.tensor([[0.5], [-1.0]]))
        assert env.transition.success is None
        assert env.transition.reward.tolist() == pytest.approx([0.1, 0.0])
        assert env.transition.terminated.tolist() == [False, True]
        assert env.transition.truncated.tolist() == [False, False]
        env.step(torch.zeros(2, 1))
        assert env.transition.terminated.tolist() == [False, True]
        assert env.transition.truncated.tolist() == [True, False]
        assert env.transition.done.all()


def test_catalog_lists_all_ocbench_tasks(fake_ocbench, monkeypatch):
    from torch_world import envs
    from torch_world.envs import get_spec, list_envs, make

    _patch_backend(monkeypatch, False)
    specs = list_envs(backend="ocbench")
    assert len(specs) == 28 and {s.category for s in specs} == {"manipulation"}
    assert all(s.id == f"OCBench/{s.kwargs['name']}" for s in specs)
    assert "OCBench" in envs.__all__ and not any("/" in name for name in envs.__all__)
    with make("OCBench/chamber-easy-task1-v0", batch_size=2) as env:
        assert get_spec(env) is get_spec("OCBench/chamber-easy-task1-v0")
        assert fake_ocbench[0].name == "chamber-cpu-easy-task1-v0"


def test_load_ocbench_variable_length_state_and_filtering(tmp_path):
    from torch.utils.data import ConcatDataset

    from torch_world.datasets import TrajectoryDataset, load_ocbench

    path = tmp_path / "state.npz"
    np.savez(
        path,
        observation_interval=np.asarray(1, dtype=np.int32),
        observations=np.arange(9 * 2, dtype=np.float32).reshape(9, 2),
        actions=np.arange(7 * 3, dtype=np.float32).reshape(7, 3),
        rewards=np.array([0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
        masks=np.array([1, 1, 0, 1, 1, 1, 1], dtype=np.float32),
        terminals=np.array([0, 0, 1, 0, 0, 0, 1], dtype=bool),
    )
    episodes = load_ocbench(path)
    assert len(episodes) == 2
    assert episodes[0]["observations"].shape == (1, 4, 2) and episodes[0]["actions"].shape == (1, 4, 3)
    assert episodes[1]["observations"].shape == (1, 5, 2) and episodes[1]["actions"].shape == (1, 5, 3)
    assert episodes[0]["actions"][0, -1].tolist() == [0, 0, 0]
    assert episodes[0]["rewards"][0].tolist() == [0, 0, 1, 0]
    assert episodes[0]["terminated"][0].tolist() == [False, False, True, False]
    assert not episodes[0]["truncated"].any()
    assert not episodes[1]["terminated"].any()
    assert episodes[1]["truncated"][0].tolist() == [False, False, False, True, False]
    windows = ConcatDataset([TrajectoryDataset(ep, window=2) for ep in episodes])
    assert len(windows) == 7

    only_success = load_ocbench(path, success_only=True)
    assert len(only_success) == 1 and only_success[0]["observations"].shape == (1, 4, 2)
    capped = load_ocbench([path, path], max_episodes=3)
    assert len(capped) == 3


def test_load_ocbench_sparse_multicamera_and_downloads(tmp_path, monkeypatch):
    from torch_world.datasets import TrajectoryDataset, load_ocbench

    path = tmp_path / "visual.npz"
    obs = np.zeros((8, 3, 8, 8, 3), dtype=np.uint8)
    obs[:, 0] = 10
    obs[:, 1] = 20
    obs[:, 2] = 30
    np.savez(
        path,
        observation_interval=np.asarray(2, dtype=np.int32),
        observations=obs,
        actions=np.arange(10 * 2, dtype=np.float32).reshape(10, 2),
        rewards=np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32),
        masks=np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=np.float32),
        terminals=np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype=bool),
    )
    front = load_ocbench(path, camera="front", size=4)
    assert len(front) == 2
    assert front[0]["observations"].shape == (1, 4, 3, 4, 4) and front[0]["observations"].dtype == torch.uint8
    assert (front[0]["observations"] == 10).all()
    assert front[0]["actions"].shape == (1, 4, 4)
    assert front[0]["actions"][0, 0].tolist() == [0, 1, 2, 3]
    assert front[0]["actions"][0, 2].tolist() == [8, 9, 0, 0]
    assert front[0]["actions"][0, 3].tolist() == [0, 0, 0, 0]
    assert front[0]["rewards"][0].tolist() == [0, 0, 1, 0]
    assert front[0]["terminated"][0].tolist() == [False, False, True, False]
    assert front[1]["truncated"][0].tolist() == [False, False, True, False]
    assert len(TrajectoryDataset(front[0], window=2)) == 3

    side = load_ocbench(path, camera="side")
    assert side[0]["observations"].shape == (1, 4, 3, 8, 8) and (side[0]["observations"] == 20).all()
    all_views = load_ocbench(path, camera=None, size=4)
    assert all_views[0]["observations"].shape == (1, 4, 3, 3, 4, 4)
    assert (all_views[0]["observations"][:, :, 2] == 30).all()

    calls = []

    def download_datasets(name, dataset_root=None, num_shards=None):
        calls.append((name, dataset_root, num_shards))
        return [str(path)], [str(path)]

    monkeypatch.setitem(sys.modules, "ocbench", SimpleNamespace(download_datasets=download_datasets))
    loaded = load_ocbench("visual-block-single-task1-v0", split="val", dataset_root=tmp_path, num_shards=1)
    assert len(loaded) == 2 and calls == [("visual-block-single-task1-v0", tmp_path, 1)]

    with pytest.raises(ValueError, match="split"):
        load_ocbench(path, split="test")
    with pytest.raises(ValueError, match="camera"):
        load_ocbench(path, camera="top")
    with pytest.raises(ValueError, match="out of range"):
        load_ocbench(path, camera=5)
    with pytest.raises(FileNotFoundError):
        load_ocbench(tmp_path / "missing.npz")
    bad = tmp_path / "bad.npz"
    np.savez(
        bad,
        observation_interval=np.asarray(1, dtype=np.int32),
        observations=np.zeros((2, 2), dtype=np.float32),
        actions=np.zeros((3, 1), dtype=np.float32),
        rewards=np.zeros(3, dtype=np.float32),
        masks=np.ones(3, dtype=np.float32),
        terminals=np.array([0, 0, 1], dtype=bool),
    )
    with pytest.raises(ValueError, match="observation boundaries"):
        load_ocbench(bad)


@pytest.mark.skipif(not HAS_OCBENCH, reason="requires the ocbench extra")
@pytest.mark.parametrize(
    ("name", "has_success"),
    [
        ("block-lite-single-task1-v0", True),
        ("chamber-lite-easy-task1-v0", True),
        ("switch-lite-3x3-task1-v0", True),
        ("hanoi-lite-single-task1-v0", True),
        ("bowling-task1-v0", False),
    ],
)
def test_real_tasks(name, has_success):
    from torch_world.envs import OCBench

    with OCBench(name, batch_size=2, render_size=32, sim_backend="cpu") as env:
        first = env.reset(seed=3)
        assert first.ndim == 2 and env.action_dim == 7
        assert torch.equal(first, env.reset(seed=3)) and not torch.equal(first[0], first[1])
        start_frame = env.render_frame()
        obs = env.step(torch.zeros(2, env.action_dim))
        assert torch.isfinite(obs).all()
        if has_success:
            assert env.transition.success.dtype == torch.bool
        else:
            assert env.transition.success is None
        qpos = [lane.unwrapped.data.qpos.copy() for lane in env.envs]
        assert env.render_frame().shape == (2, 3, 32, 32) and env.render_frame().dtype == torch.uint8
        assert torch.equal(env.render_goal_frame(), start_frame)
        assert all(np.array_equal(lane.unwrapped.data.qpos, q) for lane, q in zip(env.envs, qpos, strict=True))


@pytest.mark.skipif(not HAS_OCBENCH, reason="requires the ocbench extra")
def test_real_visual_observations_and_camera_validation():
    from torch_world.envs import OCBench

    with OCBench("visual-block-lite-single-task1-v0", size=32, render_size=64, sim_backend="cpu") as env:
        obs = env.reset(seed=1)
        assert obs.shape == (1, 3, 32, 32) and obs.dtype == torch.float32
        assert 0 <= obs.min() <= obs.max() <= 1 and env.state.ndim == 2
        assert env.render_frame().shape == (1, 3, 64, 64) and env.render_goal_frame().shape == (1, 3, 64, 64)
        batch = env.collect(2, generator=torch.Generator().manual_seed(0))
        assert batch["pixels"].shape == (1, 2, 3, 32, 32) and batch["actions"].shape == (1, 2, 7)
    with pytest.raises(ValueError, match="camera"):
        OCBench("block-lite-single-task1-v0", camera="unknown", sim_backend="cpu")
