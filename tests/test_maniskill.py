"""ManiSkill adapter: lane grouping and absorbing lanes on a fake simulator, then the real package when it can run."""

import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

HAS_MANISKILL = importlib.util.find_spec("mani_skill") is not None


class FakeEnv:
    """``num_envs`` points that succeed at x >= 1, with ManiSkill's batched tensors and per-env reset seeds."""

    def __init__(self, name, num_envs, **kwargs):
        self.name, self.num_envs, self.kwargs = name, num_envs, kwargs
        space = SimpleNamespace(low=torch.full((1,), -2.0).numpy(), high=torch.full((1,), 2.0).numpy(), shape=(1,))
        self.unwrapped = SimpleNamespace(single_action_space=space, device=torch.device("cpu"))
        self.spec = SimpleNamespace(max_episode_steps=kwargs.get("max_episode_steps", 3))
        self.actions, self.closed = [], False

    def _obs(self):
        if self.kwargs["obs_mode"] == "state":
            return self.x.clone()
        image = (self.x[:, :, None, None] * 0 + 128).expand(-1, 4, 4, 3).to(torch.uint8)
        return {"state": self.x.clone(), "sensor_data": {"base_camera": {"rgb": image}}}

    def reset(self, seed=None, options=None):
        self.seed = seed
        self.x = torch.tensor([[float(s % 7) / 10] for s in seed])
        return self._obs(), {}

    def step(self, action):
        self.actions.append(action.clone())
        self.x = self.x + action
        success = self.x[:, 0] >= 1
        return self._obs(), success.float(), success, torch.zeros_like(success), {"success": success}

    def render(self):
        return torch.full((self.num_envs, 6, 6, 3), 9, dtype=torch.uint8)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_maniskill(monkeypatch):
    made = []

    def make(name, num_envs, **kwargs):
        made.append(FakeEnv(name, num_envs, **kwargs))
        return made[-1]

    envs = ModuleType("mani_skill.envs")
    registration = ModuleType("mani_skill.utils.registration")
    registration.REGISTERED_ENVS = {"PickCube-v1": None, "Fake-v1": None}
    for name, module in {
        "gymnasium": SimpleNamespace(make=make),
        "mani_skill": ModuleType("mani_skill"),
        "mani_skill.envs": envs,
        "mani_skill.utils": ModuleType("mani_skill.utils"),
        "mani_skill.utils.registration": registration,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return made


@pytest.mark.parametrize("gpu", [False, True])
def test_lanes_absorb_after_success_on_either_backend(fake_maniskill, monkeypatch, gpu):
    from torch_world.envs import ManiSkill

    monkeypatch.setattr(torch.cuda, "is_available", lambda: gpu)
    env = ManiSkill("Fake-v1", batch_size=2)
    assert len(fake_maniskill) == (1 if gpu else 2)
    assert fake_maniskill[0].kwargs["sim_backend"] == ("physx_cuda" if gpu else "physx_cpu")
    env.reset(seed=4)
    assert [e.seed for e in fake_maniskill] == ([[4, 5]] if gpu else [[4], [5]])
    env.step(torch.tensor([[0.5], [0.0]]))  # [-1, 1] maps to the upstream bounds [-2, 2]
    assert env.state[:, 0].tolist() == pytest.approx([1.4, 0.5])
    assert env.transition.terminated.tolist() == [True, False] and env.transition.success.tolist() == [True, False]
    env.step(torch.tensor([[0.5], [0.0]]))
    assert env.state[0, 0].item() == pytest.approx(1.4)  # a finished lane keeps its last outcome
    assert env.transition.reward.tolist() == [0, 0] and env.transition.success.tolist() == [True, False]
    if gpu:
        assert fake_maniskill[0].actions[1].tolist() == [[0.0], [0.0]]  # finished lanes are stepped with zeros
    else:
        assert len(fake_maniskill[0].actions) == 1  # a finished CPU lane is not stepped again
    env.step(torch.zeros(2, 1))
    assert env.transition.truncated.tolist() == [False, True] and env.transition.done.all()
    assert env.render_frame().shape == (2, 3, 6, 6) and env.render_goal_frame().dtype == torch.uint8
    env.close()
    assert all(e.closed for e in fake_maniskill)


def test_camera_observations_and_options(fake_maniskill, monkeypatch):
    from torch_world.envs import ManiSkill, get_spec, make

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    env = make("ManiSkill/PickCube-v1", observation="rgb", size=4, control_mode="pd_ee_delta_pose")
    obs = env.reset(seed=0)
    assert obs.shape == (1, 3, 4, 4) and obs.max() == pytest.approx(128 / 255) and env.state.shape == (1, 1)
    assert fake_maniskill[0].kwargs["control_mode"] == "pd_ee_delta_pose"
    assert fake_maniskill[0].kwargs["sensor_configs"] == {"width": 4, "height": 4}
    assert get_spec(env).id == "ManiSkill/PickCube-v1"
    with pytest.raises(ValueError, match="checkpoints"):
        env.reset(torch.zeros(1, 1))
    headless = ManiSkill("Fake-v1", render_backend="none")
    assert fake_maniskill[-1].kwargs["render_mode"] is None
    with pytest.raises(RuntimeError, match="render_backend"):
        headless.render_frame()
    with pytest.raises(ValueError, match="render backend"):
        ManiSkill("Fake-v1", observation="rgb", render_backend="none")
    patient = ManiSkill("Fake-v1", batch_size=2, terminate_on_success=False)
    patient.reset(seed=4)
    patient.step(torch.tensor([[0.5], [0.0]]))
    assert patient.transition.success.tolist() == [True, False] and not patient.transition.terminated.any()
    with pytest.raises(ValueError, match="Unknown ManiSkill"):
        ManiSkill("PickSphere-v9")


def _maniskill_runs():
    """ManiSkill's CPU simulation without rendering, which macOS cannot select: it always renders there."""
    return HAS_MANISKILL and sys.platform.startswith("linux")


@pytest.mark.skipif(not _maniskill_runs(), reason="requires the maniskill extra on Linux")
def test_real_task():
    from torch_world.envs import make

    with make("ManiSkill/PickCube-v1", batch_size=2, render_backend="none", sim_backend="physx_cpu") as env:
        first = env.reset(seed=3)
        assert first.ndim == 2 and env.max_episode_steps == 50
        assert torch.equal(first, env.reset(seed=3)) and not torch.equal(first[0], first[1])
        obs = env.step(torch.zeros(2, env.action_dim))
        assert torch.isfinite(obs).all() and env.transition.success.dtype == torch.bool
