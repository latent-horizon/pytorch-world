"""The shared protocol of the external MuJoCo suites, on a one-joint MuJoCo model; real suites run when installed."""

import importlib.util

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco")

from torch_world.envs.lanes import MujocoLanes

SLIDER = """
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <camera name="front" pos="0 -1 0.2" xyaxes="1 0 0 0 0 1"/>
    <body name="cart"><joint name="x" type="slide" axis="1 0 0"/><geom type="box" size="0.05 0.05 0.05"/></body>
  </worldbody>
</mujoco>
"""


class Slider:
    """A cart on a rail that succeeds at x >= 0.1 and times out after 4 steps, with a seeded initial position."""

    def __init__(self, mujoco):
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(SLIDER)
        self.data = mujoco.MjData(self.model)
        self.steps, self.closed = 0, False

    def reset(self, seed):
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = np.random.default_rng(seed).uniform(-0.05, 0)
        self.mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        return self.data.qpos.copy()

    def step(self, action):
        self.data.qpos[0] += action[0]
        self.mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        return self.data.qpos.copy(), float(action[0]), self.steps >= 4

    def close(self):
        self.closed = True


class SliderLanes(MujocoLanes):
    suite = "Slider"

    def __init__(self, batch_size=2, observation="state", success=True):
        super().__init__(batch_size, 16, None, observation=observation, camera="front", quality="low", render_size=24)
        self.has_success = success
        self._build(lambda: Slider(self.mujoco))
        self._bounds([-0.2], [0.2])
        self.max_episode_steps = 10
        self.stepped = [0] * batch_size

    def _physics(self, lane):
        return self.envs[lane].model, self.envs[lane].data

    def _reset_lane(self, lane, seed):
        return self.envs[lane].reset(seed)

    def _step_lane(self, lane, action):
        self.stepped[lane] += 1
        state, reward, timeout = self.envs[lane].step(action)
        success = bool(state[0] >= 0.1)
        return state, reward, success, timeout, success if self.has_success else None


def test_actions_are_rescaled_and_finished_lanes_are_absorbing():
    env = SliderLanes()
    env.reset(seed=0)
    start = env.state.clone()
    env.step(torch.tensor([[1.0], [0.0]]))  # [-1, 1] maps to the upstream bounds [-0.2, 0.2]
    assert torch.allclose(env.state[:, 0] - start[:, 0], torch.tensor([0.2, 0.0]))
    assert env.transition.terminated.tolist() == [True, False] and env.transition.success.tolist() == [True, False]
    assert env.transition.reward.tolist() == pytest.approx([0.2, 0.0])
    for _ in range(3):
        env.step(torch.ones(2, 1) * 0.1)
    assert env.stepped == [1, 4]  # the finished lane is never stepped again
    assert env.transition.truncated.tolist() == [False, True] and env.transition.done.all()
    assert env.transition.success.tolist() == [True, False] and env.transition.reward[0] == 0
    env.close()
    assert env.envs == []


def test_seeded_resets_and_missing_success():
    env = SliderLanes(success=False)
    first = env.reset(seed=3)
    assert torch.equal(first, env.reset(seed=3)) and not torch.equal(first[0], first[1])
    assert not torch.equal(first, env.reset(seed=4))
    env.step(torch.ones(2, 1))
    assert env.transition.success is None  # a suite without a success signal never reports failure
    with pytest.raises(ValueError, match="checkpoints"):
        env.reset(torch.zeros(2, 1))
    with pytest.raises(ValueError, match="Arbitrary-state"):
        env.render(env.state)


def test_pixel_observations_and_reset_frames_leave_the_simulation_untouched():
    env = SliderLanes(observation="rgb")
    obs = env.reset(seed=0)
    assert obs.shape == (2, 3, 16, 16) and obs.dtype == torch.float32 and 0 <= obs.min() and obs.max() <= 1
    start = env.render_frame()
    assert start.shape == (2, 3, 24, 24) and start.dtype == torch.uint8
    env.step(torch.tensor([[0.5], [-0.5]]))
    moved = env.render_frame()
    qpos = [lane.data.qpos.copy() for lane in env.envs]
    assert torch.equal(env.render_goal_frame(), start)
    assert torch.equal(env.render_frame(), moved)
    assert all(np.array_equal(lane.data.qpos, q) for lane, q in zip(env.envs, qpos, strict=True))
    assert not torch.equal(env.render(), obs)
    env.close()


@pytest.mark.skipif(importlib.util.find_spec("dm_control") is None, reason="requires the dmcontrol extra")
def test_dmcontrol():
    from torch_world.envs import DMControl, get_spec, make

    with make("DMControl/cartpole-swingup", batch_size=2, action_repeat=2, render_size=32) as env:
        first = env.reset(seed=1)
        assert first.shape == (2, 5) and env.action_dim == 1 and env.max_episode_steps == 500
        assert torch.equal(first, env.reset(seed=1)) and not torch.equal(first[0], first[1])
        env.step(torch.zeros(2, 1))
        assert env.transition.success is None and (env.transition.reward >= 0).all()
        assert env.render_frame().shape == (2, 3, 32, 32) and get_spec(env).category == "control"
    with DMControl("walker-walk", observation="rgb") as env:
        assert env.reset(seed=0).shape == (1, 3, 64, 64) and env.state.shape == (1, 24)
    with pytest.raises(ValueError, match="domain-task"):
        DMControl("cheetah-fly")


@pytest.mark.skipif(importlib.util.find_spec("metaworld") is None, reason="requires the metaworld extra")
def test_metaworld():
    from torch_world.envs import MetaWorld, make

    with make("MetaWorld/reach-v3", batch_size=2, render_size=32) as env:
        first = env.reset(seed=1)
        np.random.seed(0)  # placements come from each lane's own generator, not NumPy's global one
        assert first.shape == (2, 39) and env.action_dim == 4 and env.max_episode_steps == 500
        assert torch.equal(first, env.reset(seed=1)) and not torch.equal(first[0, -3:], first[1, -3:])
        env.step(torch.zeros(2, 4))
        targets = torch.tensor(np.stack([lane._target_pos for lane in env.envs]), dtype=torch.float32)
        assert torch.allclose(env.state[:, -3:], targets)  # the goal stays observed after steps, not clipped away
        assert env.transition.success.dtype == torch.bool and env.render_goal_frame().shape == (2, 3, 32, 32)
    with MetaWorld("pick-place-v3", observation="rgb", size=32) as env:
        assert env.reset(seed=0).shape == (1, 3, 32, 32) and env.flip
    with pytest.raises(ValueError, match="pick-place-v3"):
        MetaWorld("reach-v2")


def _myosuite_builds():
    if importlib.util.find_spec("myosuite") is None:
        return False
    import mujoco

    return tuple(map(int, mujoco.__version__.split(".")[:2])) >= (3, 7)


@pytest.mark.skipif(not _myosuite_builds(), reason="requires the myosuite extra")
def test_myosuite():
    from torch_world.envs import MyoSuite, make

    # Not myoFingerReachRandom: MyoSuite 3.0 samples its targets from a workspace that collapses to one point.
    with make("MyoSuite/myoFingerPoseRandom-v0", batch_size=2, render_size=32) as env:
        first = env.reset(seed=1)
        assert first.ndim == 2 and env.max_episode_steps == 100 and env.terminate_on_success
        assert torch.equal(first, env.reset(seed=1)) and not torch.equal(first[0], first[1])
        env.step(torch.zeros(2, env.action_dim))
        assert env.transition.success.dtype == torch.bool and env.render_frame().shape == (2, 3, 32, 32)
    with MyoSuite("myoLegWalk-v0", observation="rgb", size=32) as env:
        assert env.camera == "side_view" and not env.terminate_on_success
        assert env.reset(seed=0).shape == (1, 3, 32, 32)
