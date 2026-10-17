"""Real MuJoCo replay and rejection evidence, independent of learned weights."""

import numpy as np
import pytest
import torch

from torch_world.sim2real.config import RandomizationConfig
from torch_world.sim2real.control import ClearanceError
from torch_world.sim2real.investigation import action_tracking, trace_episode
from torch_world.sim2real.models import make_adapter
from torch_world.sim2real.reference import reference_setup
from torch_world.sim2real.simulation import SO101Chess

pytest.importorskip("mujoco")


def test_fixed_task_requested_and_applied_replay_match():
    config, task = reference_setup(size=16)
    with SO101Chess(config, task, randomization=RandomizationConfig(enabled=False)) as env:
        settings = {"seed": 10007, "source": "b2", "target": "a5"}
        demo, report = trace_episode(env, **settings)
        assert report["outcome"]["success"] and report["rejection"] is None
        for actions in (demo.requested_actions, demo.actions):
            replay, result = trace_episode(env, **settings, replay=actions)
            assert result["outcome"]["success"]
            torch.testing.assert_close(replay.proprio, demo.proprio, rtol=0, atol=0)
        assert len(demo.pixels) == len(demo.actions) + 1
        adapter = make_adapter("lewam", size=16, smoke=True, proprioception=True, chunk_size=8).eval()
        weights = {k: v.clone() for k, v in adapter.state_dict().items()}
        tracking = action_tracking(adapter, demo)
        assert tracking["active_windows"] > 0
        assert np.isfinite(tracking["first_action_normalized_mse"])
        for key, value in adapter.state_dict().items():
            torch.testing.assert_close(value, weights[key])


def test_clearance_evidence_does_not_advance_physics_or_update_target():
    config, task = reference_setup(size=16)
    with SO101Chess(config, task, randomization=RandomizationConfig(enabled=False)) as env:
        env.reset(seed=10007, source="b2", target="a5")
        for _ in range(40):
            time, target = env.data.time, env.controller.target.copy()
            positions = env.data.qpos.copy()
            try:
                env.step([0, 0, -1, 0])
            except ClearanceError as error:
                assert env.data.time == time
                np.testing.assert_array_equal(env.data.qpos, positions)
                np.testing.assert_array_equal(env.controller.target, target)
                assert error.contacts and any("world" in c["bodies"] for c in error.contacts)
                assert all(c["distance_m"] < -0.001 for c in error.contacts)
                assert all(0 <= c["fraction"] <= 1 for c in error.contacts)
                contacts = env.controller.clearance_contacts(
                    error.start, error.target, error.opening, start_opening=env.opening
                )
                assert contacts
                np.testing.assert_array_equal(env.data.qpos, positions)
                break
        else:
            pytest.fail("Expected the downward command to be rejected at the board")
