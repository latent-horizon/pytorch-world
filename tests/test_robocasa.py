"""RoboCasa adapter and dataset loader: protocol tests on a fake kitchen, then the real package when installed."""

import importlib.util
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

HAS_ROBOCASA = importlib.util.find_spec("robocasa") is not None


class FakeKitchen:
    """Moves a gripper coordinate by the first action; success at 1. Resets draw the layout from env.rng and the
    color from np.random, both in the proprioception here so that tests can see them."""

    def __init__(self, task, cameras):
        self.task, self.cameras, self.closed = task, cameras, False
        self.action_spec = (-np.ones(12), np.ones(12))
        self.rng = np.random.default_rng(0)
        self.steps = 0

    def observe(self):
        images = {}
        for camera, size in self.cameras.items():
            images[f"{camera}_image"] = np.zeros((size, size, 3), np.uint8)
            images[f"{camera}_image"][-1] = 255  # the bottom row as stored is the top row once flipped upright
        return {"robot0_proprio-state": np.array([self.x + self.color / 1000, self.layout]), **images}

    def reset(self):
        self.x, self.layout, self.color = 0.0, float(self.rng.integers(100)), float(np.random.randint(100))
        return self.observe()

    def step(self, action):
        self.steps += 1
        self.x += float(action[0])
        return self.observe(), 0.0, False, {}

    def _check_success(self):
        return self.x >= 1

    def get_ep_meta(self):
        return getattr(self, "ep_meta", {"lang": f"do {self.task}"})

    # RoboCasa's playback restore: episode metadata, the scene's MJCF, then the flattened simulator state.
    def set_ep_meta(self, meta):
        self.ep_meta = meta

    def edit_model_xml(self, xml):
        return xml

    def reset_from_xml_string(self, xml):
        self.xml = xml

    @property
    def sim(self):
        return SimpleNamespace(reset=lambda: None, forward=lambda: None, set_state_from_flattened=self._set_state)

    def _set_state(self, state):
        self.x, self.layout = float(state[0]), float(state[1])

    def update_state(self):
        pass

    def _get_observations(self, force_update=False):
        return self.observe()

    def close(self):
        self.closed = True


@pytest.fixture
def fake_robocasa(monkeypatch):
    made = []

    def create_env(task, robots, camera_names, camera_widths, camera_heights, split, seed, **kwargs):
        assert camera_widths == camera_heights
        made.append(FakeKitchen(task, dict(zip(camera_names, camera_widths))))
        return made[-1]

    modules = {
        "robocasa": ModuleType("robocasa"),
        "robocasa.utils": ModuleType("robocasa.utils"),
        "robocasa.utils.env_utils": SimpleNamespace(create_env=create_env),
        "robocasa.utils.dataset_registry_utils": SimpleNamespace(get_task_horizon=lambda task: 3),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return made


def test_success_ends_lanes_and_frames_are_upright(fake_robocasa, monkeypatch):
    from torch_world.envs import RoboCasa, get_spec

    monkeypatch.setattr(sys, "platform", "linux")

    env = RoboCasa("OpenDrawer", batch_size=2, size=4, render_size=8)
    obs = env.reset(seed=0)
    assert obs.shape == (2, 3, 4, 4) and obs[:, :, 0].eq(1).all() and obs[:, :, -1].eq(0).all()
    assert env.instruction == ["do OpenDrawer"] * 2 and env.gt_label == "Task · do OpenDrawer"
    assert get_spec(env).id == "RoboCasa/OpenDrawer" and env.max_episode_steps == 3 and env.state.shape == (2, 2)
    env.step(torch.tensor([[1.0] + [0] * 11, [0.0] * 12]))
    assert env.transition.success.tolist() == [True, False] and env.transition.reward.tolist() == [1, 0]
    assert env.transition.terminated.tolist() == [True, False]
    env.step(torch.zeros(2, 12))
    env.step(torch.zeros(2, 12))
    assert fake_robocasa[0].steps == 1  # a finished lane is not stepped again
    assert env.transition.truncated.tolist() == [False, True] and env.transition.done.all()
    assert fake_robocasa[0].cameras == {"robot0_agentview_left": 4, "robot0_agentview_center": 8}
    assert env.render_frame().shape == (2, 3, 8, 8) and env.render_goal_frame().dtype == torch.uint8
    assert env.render_frame()[:, :, 0].eq(255).all()  # presentation frames are upright too
    env.close()
    assert all(k.closed for k in fake_robocasa)


def test_macos_refuses_batched_pixels(fake_robocasa, monkeypatch):
    from torch_world.envs import RoboCasa

    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(ValueError, match="GLFW"):
        RoboCasa("OpenDrawer", batch_size=2)
    RoboCasa("OpenDrawer", batch_size=2, observation="state").close()
    RoboCasa("OpenDrawer", batch_size=1).close()


def test_seeded_resets_reproduce_the_kitchen_and_spare_the_callers_generator(fake_robocasa):
    from torch_world.envs import RoboCasa

    env = RoboCasa("OpenDrawer", batch_size=2, size=4, observation="state")
    np.random.seed(5)
    outer = np.random.get_state()[1].copy()
    first = env.reset(seed=3)
    np.random.uniform()
    assert torch.equal(first, env.reset(seed=3)) and not torch.equal(first[0], first[1])
    np.random.seed(5)
    assert np.array_equal(outer, np.random.get_state()[1])
    assert not torch.equal(first, env.reset(seed=4))
    with pytest.raises(ValueError, match="checkpoints"):
        env.reset(torch.zeros(2, 2))


def test_scenes_restore_recorded_demonstrations(fake_robocasa):
    from torch_world.envs import RoboCasa

    env = RoboCasa("OpenDrawer", batch_size=1, size=4, observation="state")
    env.reset(scenes=[{"ep_meta": {"lang": "open it"}, "model": "<mujoco/>", "state": np.array([0.5, 42.0])}])
    assert env.instruction == ["open it"] and fake_robocasa[0].xml == "<mujoco/>"
    assert env.state[0, 1] == 42 and abs(env.state[0, 0].item() - 0.5) < 0.1
    with pytest.raises(ValueError, match="one scene per lane"):
        env.reset(scenes=[])


def test_catalog_lists_the_target_tasks():
    from torch_world.envs import list_envs

    specs = list_envs(backend="robocasa")
    assert len(specs) == 50 and {s.category for s in specs} == {"manipulation"}
    assert sum("atomic" in s.tags for s in specs) == 18 and sum("unseen" in s.tags for s in specs) == 16


def test_load_robocasa_reorders_actions_to_the_environment(tmp_path):
    av = pytest.importorskip("av")
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from torch_world.datasets import load_robocasa

    T = 6
    info = {
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    fields = {"base_motion": (0, 4), "control_mode": (4, 5), "end_effector_position": (5, 8),
              "end_effector_rotation": (8, 11), "gripper_close": (11, 12)}  # fmt: skip
    modality = {"action": {k: {"start": a, "end": b} for k, (a, b) in fields.items()}}
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(json.dumps(info))
    (tmp_path / "meta" / "modality.json").write_text(json.dumps(modality))
    stored = np.tile(np.arange(12, dtype=np.float64), (T, 1))
    table = pa.table({"action": list(stored), "observation.state": list(np.zeros((T, 16)))})
    (tmp_path / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(table, tmp_path / "data" / "chunk-000" / "episode_000002.parquet")
    video = tmp_path / "videos" / "chunk-000" / "observation.images.robot0_agentview_left" / "episode_000002.mp4"
    video.parent.mkdir(parents=True)
    with av.open(str(video), "w") as container:
        stream = container.add_stream("libx264", rate=20)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for _ in range(T):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), 128, np.uint8), format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    episode = load_robocasa(tmp_path, 2, size=8)
    assert episode["observations"].shape == (1, T, 3, 8, 8) and episode["states"].shape == (1, T, 16)
    # environment order: end effector position, rotation, gripper, base motion, control mode
    assert episode["actions"][0, 0].tolist() == [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]


@pytest.mark.skipif(not HAS_ROBOCASA, reason="requires RoboCasa, installed from source with its assets")
def test_real_task():
    from torch_world.envs import make

    with make("RoboCasa/OpenDrawer", batch_size=1, size=64, render_size=64) as env:
        obs = env.reset(seed=0)
        assert obs.shape == (1, 3, 64, 64) and env.action_dim == 12 and env.instruction[0]
        assert torch.isfinite(env.step(torch.zeros(1, 12))).all() and env.transition.success is not None
