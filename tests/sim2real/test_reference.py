from copy import deepcopy

import pytest
import torch

from torch_world.sim2real import Episode, EpisodeWindows, make_adapter, reference_setup, simulation_gate
from torch_world.sim2real.data import cache_visual_features


def test_reference_dimensions_and_unverified_status():
    config, task = reference_setup()
    assert 8 * task.square_size + 2 * task.appearance.border == pytest.approx(0.25)
    assert task.appearance.surface == 0.001
    assert not task.measured
    assert config.wrist_view.aspect == pytest.approx(16 / 9)


def test_cached_features_preserve_loss_and_trainable_gradients():
    torch.manual_seed(3)
    adapter = make_adapter("dinowm", size=16, smoke=True)
    episode = Episode(
        torch.randint(256, (5, 3, 16, 16), dtype=torch.uint8), torch.rand(5, 6), torch.rand(4, 4), torch.rand(4, 4), {}
    )
    raw = EpisodeWindows([episode], 3)[0]
    cached = EpisodeWindows([episode], 3, cache_visual_features(adapter, [episode]))[0]
    a = adapter.loss(raw["pixels"][None], raw["actions"][None], raw["states"][None])["loss"]
    b = adapter.loss_features(cached["features"][None], cached["actions"][None], cached["states"][None])["loss"]
    torch.testing.assert_close(a, b)
    b.backward()
    assert adapter.model.proprio_encoder.weight.grad.abs().sum() > 0
    assert adapter.model.action_encoder.weight.grad.abs().sum() > 0


def test_gate_uses_learned_heldout_success_and_latency():
    result = {"smoke": False, "config": {"control_dt": 0.1}, "evaluation": {}}
    for domain in ("nominal", "randomized"):
        result["evaluation"][domain] = {
            "learned": {
                "episodes": [{"success": True, "max_planning_seconds": 0.02, "failure": None} for _ in range(24)]
            }
        }
    assert simulation_gate(result)["passed"]
    for latency in (float("nan"), float("inf"), -0.1):
        invalid = deepcopy(result)
        invalid["evaluation"]["nominal"]["learned"]["episodes"][0]["max_planning_seconds"] = latency
        assert not simulation_gate(invalid)["passed"]
    bad = deepcopy(result)
    for r in bad["evaluation"]["randomized"]["learned"]["episodes"]:
        r["success"] = False
    bad["evaluation"]["randomized"]["scripted"] = {"success_rate": 1.0}
    assert not simulation_gate(bad)["passed"]
    result["evaluation"]["nominal"]["learned"]["episodes"][0]["max_planning_seconds"] = 0.2
    assert not simulation_gate(result)["passed"]
    assert not simulation_gate(result)["physical_trial_enabled"]


def test_reference_native_view_and_model_observations():
    pytest.importorskip("mujoco")
    from torch_world.sim2real import SO101Chess

    config, task = reference_setup(size=32)
    with SO101Chess(config, task) as env:
        observation = env.reset(source="a4", target="b4")
        assert observation.rgb.shape == (3, 32, 32)
        assert env.render_view(size=128, native=True).shape == (3, 72, 128)
        assert env.goal(size=128, native=True).rgb.shape == (3, 72, 128)
        assert env.observe().rgb.shape == (3, 32, 32)


def test_reference_preview_restores_episode_and_camera_mount():
    pytest.importorskip("mujoco")
    import numpy as np

    from torch_world.sim2real import SO101Chess, render_reference_pose

    config, task = reference_setup(size=32)
    with SO101Chess(config, task) as env:
        env.reset(source="a4", target="b4")
        # The corrected camera is on the actual mount's side of the jaws.
        np.testing.assert_allclose(env.model.camera("wrist").pos, env.model.camera("wrist_cam").pos, atol=1e-7)
        before = env.observe().rgb.clone()
        kind = env.mj.mjtState.mjSTATE_INTEGRATION
        state = np.empty(env.mj.mj_stateSize(env.model, kind))
        env.mj.mj_getState(env.model, env.data, state, kind)
        preview = render_reference_pose(env, size=640)
        assert preview.shape == (3, 360, 640)
        restored = np.empty_like(state)
        env.mj.mj_getState(env.model, env.data, restored, kind)
        np.testing.assert_array_equal(state, restored)
        torch.testing.assert_close(env.observe().rgb, before)


def test_rest_view_projects_the_photographed_board(monkeypatch):
    """Guard perspective and placement, not just the gripper's image height."""
    pytest.importorskip("mujoco")
    import json
    from pathlib import Path

    import numpy as np

    from torch_world.sim2real import SO101Chess, render_reference_pose

    report = json.loads((Path(__file__).parents[2] / "examples/sim2real/assets/rest-view-fit.json").read_text())
    config, task = reference_setup(size=32)
    with SO101Chess(config, task) as env:
        env.reset(source="a4", target="b4")
        projected = []
        original = env.render_view

        def capture(**kwargs):
            border, square = task.appearance.border, task.square_size
            front = np.asarray(task.board_origin) + [-border - square / 2, 3.5 * square, 0]
            points = (
                np.array(
                    [[border + row * square, 0.125 - border - col * square, 0] for row, col in report["grid_landmarks"]]
                    + [[0, 0.125, 0], [0, -0.125, 0]]
                )
                + front
            )
            camera = env.model.camera("wrist").id
            rotation = env.data.cam_xmat[camera].reshape(3, 3)
            camera_points = (points - env.data.cam_xpos[camera]) @ rotation
            focal = 360 / np.tan(np.deg2rad(config.wrist_view.fovy) / 2)
            projected.extend(camera_points[:, :2] / -camera_points[:, 2, None] * [focal, -focal] + [640, 360])
            return original(**kwargs)

        monkeypatch.setattr(env, "render_view", capture)
        render_reference_pose(env, size=128)
        assert np.sqrt(np.mean((np.array(projected) - report["observed_pixels"]) ** 2)) < 5.0


def test_fitted_profile_completes_a_scripted_contact_transfer():
    pytest.importorskip("mujoco")
    from torch_world.sim2real import SO101Chess

    config, task = reference_setup(size=16)
    with SO101Chess(config, task) as env:
        env.reset(source="a4", target="b4", seed=7)
        for _ in range(task.max_steps):
            env.step(env.expert())
            if env.done:
                break
        assert env.info()["success"]
        assert env.info()["lifted"]
