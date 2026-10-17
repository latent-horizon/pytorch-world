import numpy as np
import pytest
import torch

pytest.importorskip("mujoco")

from torch_world.sim2real import RandomizationConfig, SO101Chess, SO101Config
from torch_world.sim2real.config import PIECES


def test_calibrated_render_uses_native_camera_aspect(calibration):
    from dataclasses import replace

    calibration = replace(calibration, resolution=(64, 32), intrinsics=(32, 32, 32, 16))
    with SO101Chess(SO101Config(size=32), calibration=calibration) as env:
        env.mj.mj_forward(env.model, env.data)
        assert env.render_view(size=64, native=True).shape == (3, 32, 64)
        assert env.render_view(size=64).shape == (3, 64, 64)


@pytest.mark.parametrize("piece", PIECES)
@pytest.mark.parametrize("color", ["white", "black"])
def test_expert_releases_upright_piece_inside_square(piece, color):
    with SO101Chess(SO101Config(size=16), piece=piece, color=color) as env:
        obs = env.reset(seed=7, source="c4", target="d4")
        assert obs.rgb.shape == (3, 16, 16)
        for _ in range(env.task.max_steps):
            env.step(env.expert())
            if env.done:
                break
        assert env.info()["lifted"] and env.info()["success"], env.info()
        assert env.opening > 0.8
        with pytest.raises(RuntimeError, match="ended"):
            env.step(torch.zeros(4))


def test_seeded_reset_goal_isolation_and_randomization():
    with SO101Chess(SO101Config(size=16), randomization=RandomizationConfig()) as env:
        first = env.reset(seed=23)
        params = dict(env.episode_parameters)
        state = env.data.qpos.copy()
        goal = env.goal()
        np.testing.assert_array_equal(state, env.data.qpos)
        assert goal.rgb.shape == first.rgb.shape
        assert env.goal(size=128).rgb.shape == (3, 128, 128)
        np.testing.assert_array_equal(state, env.data.qpos)
        assert env.config.size == 16
        second = env.reset(seed=23)
        assert params == env.episode_parameters
        torch.testing.assert_close(first.rgb, second.rgb)
        np.testing.assert_array_equal(state, env.data.qpos)
        env.reset(seed=24)
        assert params != env.episode_parameters
        with pytest.raises(ValueError, match="distinct reachable"):
            env.reset(source="h8", target="h8")


def test_tipping_holding_and_transient_placement_are_not_success():
    with SO101Chess(SO101Config(size=16)) as env:
        env.reset(source="c4", target="d4")
        i = env.piece_qpos
        env.data.qpos[i : i + 3] = env.target_position
        env.mj.mj_forward(env.model, env.data)
        assert env.info()["placed"] and not env.info()["success"]
        env.data.qpos[i + 3 : i + 7] = [np.sqrt(0.5), np.sqrt(0.5), 0, 0]
        env.mj.mj_forward(env.model, env.data)
        assert not env.info()["placed"]
        env.data.qpos[i + 3 : i + 7] = [1, 0, 0, 0]
        env.data.qpos[env.grip_joint] = 0
        env.mj.mj_forward(env.model, env.data)
        assert not env.info()["placed"]
