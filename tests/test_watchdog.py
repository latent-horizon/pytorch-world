import json
import os
import time

import pytest

from torch_world.envs.mujoco import MujocoEnv, close_renderers
from torch_world.evaluation import run_episodes

# Episodes run in a spawned worker, so they live at the top level of this module.


def square(seed):
    return {"value": seed * seed, "pid": os.getpid()}


def stall_on_two(seed):
    if seed == 2:
        time.sleep(60)
    return {"pid": os.getpid()}


def fail_on_one(seed):
    if seed == 1:
        raise RuntimeError("bad episode")
    return {}


def crash_on_one(seed):
    if seed == 1:
        os._exit(3)
    return {}


def test_episodes_share_one_worker():
    records = run_episodes(square, [1, 2, 3], timeout=60)
    assert [r["status"] for r in records] == ["ok"] * 3
    assert [r["value"] for r in records] == [1, 4, 9]
    assert len({r["pid"] for r in records}) == 1 and records[0]["pid"] != os.getpid()


def test_a_stalled_episode_is_recorded_and_the_rest_continue(tmp_path):
    path = tmp_path / "episodes.jsonl"
    records = run_episodes(stall_on_two, [1, 2, 3], timeout=5, path=path)
    assert [r["status"] for r in records] == ["ok", "timeout", "ok"]
    assert records[0]["pid"] != records[2]["pid"]  # the stalled worker was replaced
    assert [json.loads(line)["seed"] for line in path.read_text().splitlines()] == [1, 2, 3]


def test_errors_and_crashes_are_recorded():
    assert [r["status"] for r in run_episodes(fail_on_one, [0, 1, 2], timeout=30)] == ["ok", "error", "ok"]
    assert [r["status"] for r in run_episodes(crash_on_one, [0, 1, 2], timeout=30)] == ["ok", "crashed", "ok"]


def test_recorded_seeds_are_skipped(tmp_path):
    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps({"seed": 1, "status": "ok", "value": -1}) + "\n")
    records = run_episodes(square, [1, 2], timeout=60, path=path)
    assert [r["value"] for r in records] == [-1, 4]
    assert len(path.read_text().splitlines()) == 2


class FakeRenderer:
    created, closed = 0, 0

    def __init__(self, model, height, width):
        FakeRenderer.created += 1
        self.scene = type("Scene", (), {"flags": {}})()

    def update_scene(self, data, camera, scene_option=None):
        pass

    def render(self):
        import numpy as np

        return np.zeros((4, 4, 3), dtype=np.uint8)

    def close(self):
        FakeRenderer.closed += 1


@pytest.fixture
def fake_env(monkeypatch):
    mujoco = pytest.importorskip("mujoco")
    monkeypatch.setattr(mujoco, "Renderer", FakeRenderer)
    FakeRenderer.created = FakeRenderer.closed = 0
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='0.1'/></worldbody></mujoco>")
    return MujocoEnv(model, batch_size=2, size=4, camera=-1)


def test_cameras_share_one_renderer_per_size(fake_env):
    mujoco = fake_env.mujoco
    for _ in range(50):
        fake_env.render(camera=mujoco.MjvCamera())  # a new camera object every call, as overview videos do
        fake_env.render(camera=-1)
    assert FakeRenderer.created == 1
    fake_env.render(size=8)
    assert FakeRenderer.created == 2


def test_close_frees_every_renderer(fake_env):
    with fake_env:
        fake_env.render()
        fake_env.render(size=8)
    assert FakeRenderer.closed == 2 and fake_env.renderers == {}
    fake_env.render()  # rendering after close makes a new renderer
    assert FakeRenderer.created == 3


def test_close_renderers_empties_the_dict():
    renderers = {1: FakeRenderer(None, 1, 1), 2: FakeRenderer(None, 1, 1)}
    FakeRenderer.closed = 0
    close_renderers(renderers)
    assert renderers == {} and FakeRenderer.closed == 2
