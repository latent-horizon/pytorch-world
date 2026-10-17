from dataclasses import replace

import numpy as np
import pytest
import torch

from torch_world.sim2real import RandomizationConfig, SO101Chess, reference_setup
from torch_world.sim2real.corrections import (
    ExpertTakeover,
    OraclePolicy,
    _trace,
    evaluation_cases,
    make_oracle,
    oracle_observation,
)
from torch_world.sim2real.data import LeWAMWindows
from torch_world.sim2real.experiment import train


def test_oracle_has_no_camera_dependence_and_requires_simulator_state():
    pytest.importorskip("mujoco")
    config, task = reference_setup(size=16)
    with SO101Chess(config, task, randomization=RandomizationConfig(enabled=False)) as env:
        observation = env.reset(seed=10007, source="b2", target="a5")
        before = env.data.qpos.copy()
        observed = oracle_observation(env, observation)
        env.phase, env.phase_steps = 7, 99
        torch.testing.assert_close(oracle_observation(env, observation).privileged, observed.privileged, rtol=0, atol=0)
        env.phase, env.phase_steps = 0, 0
        goal = oracle_observation(env, env.goal(), goal=True)
        np.testing.assert_array_equal(before, env.data.qpos)
        assert observed.privileged.shape == (28,)
        torch.testing.assert_close(goal.privileged[13:16], torch.tensor(env.target_position).float())
        adapter = make_oracle(17).eval()
        with pytest.raises(ValueError, match="simulation-only"):
            adapter.observation_state(observation)
        state = observed.privileged[None, None]
        torch.testing.assert_close(
            adapter.encode(observed.rgb[None, None], state),
            adapter.encode(torch.rand_like(observed.rgb)[None, None], state),
            rtol=0,
            atol=0,
        )
        policy = OraclePolicy(adapter, env, env.goal(), seed=17, noise=0)
        action = policy(observation)
        assert action.shape == (4,) and torch.isfinite(action).all()
        policy.record_action(action)


def test_takeover_preserves_prefix_and_masks_imitation():
    pytest.importorskip("mujoco")
    config, task = reference_setup(size=16)
    settings = {"seed": 10007, "source": "b2", "target": "a5"}
    with SO101Chess(config, task, randomization=RandomizationConfig(enabled=False)) as env:
        reference, report = _trace(env, **settings)
        assert report["outcome"]["success"]
        prefix = reference.requested_actions[:8]
        takeover = ExpertTakeover(env, prefix, report["trace"])
        corrected, result = _trace(env, **settings, policy_factory=lambda goal: takeover)
    assert result["outcome"]["success"]
    torch.testing.assert_close(corrected.proprio[:9], reference.proprio[:9], rtol=0, atol=1e-6)
    corrected.action_valid[:8] = 0
    windows = LeWAMWindows([corrected], 16, 8)
    assert not windows[0]["action_valid"].any()
    assert windows[8]["action_valid"].all()
    assert corrected.proprio.shape[-1] == 28
    assert np.isfinite(train(make_oracle(17), [corrected], updates=2, batch_size=2)).all()
    failed = replace(corrected, metadata={"outcome": {"success": False}})
    assert not LeWAMWindows([failed], 16, 8)[8]["action_valid"].any()
    assert {c["seed"] for c in evaluation_cases()}.isdisjoint(
        61000 + seed * 10 + repeat for seed in (17, 29, 43) for repeat in range(4)
    )


def test_summary_requires_complete_matching_matrix(tmp_path):
    from torch_world.sim2real.corrections import summarize
    from torch_world.sim2real.curriculum import write_json

    prepared, models = tmp_path / "prepared", tmp_path / "models"
    prepared.mkdir()
    initial = tmp_path / "oracle_initial/seed17"
    initial.mkdir(parents=True)
    write_json(initial / "training.json", {"dataset_sha256": "test", "updates": 2000})
    protocol = {
        "dataset_sha256": "test",
        "updates": 2000,
        "model_seeds": [17],
        "variants": ["image_control"],
        "noise_scales": [0.0],
        "policy_seeds": [101],
        "evaluation_cases": [{"name": "nominal"}, {"name": "disturbance_4"}],
    }
    write_json(prepared / "protocol.json", protocol)
    folder = models / "image_control-seed17"
    folder.mkdir(parents=True)
    training = {"variant": "image_control", "seed": 17, "dataset_sha256": "test", "updates": 2000}
    write_json(folder / "training.json", training)
    (folder / "model.pt").write_bytes(b"test")
    trials = [
        {
            "case": case["name"],
            "noise_scale": 0.0,
            "policy_seed": 101,
            "outcome": {"success": False, "lifted": True},
            "failure": None,
            "trace": [1],
        }
        for case in protocol["evaluation_cases"]
    ]
    evaluation = {"variant": "image_control", "seed": 17, "trials": trials}
    write_json(folder / "evaluation.json", evaluation)
    result = summarize(prepared, models, tmp_path / "summary.json")
    assert result["complete"] and result["summary"][0]["success_by_model_seed"] == {"17": 0}
    assert "trace" not in result["runs"][0]["trials"][0]
    for invalid in (trials[:1], trials[:1] * 2):
        write_json(folder / "evaluation.json", {**evaluation, "trials": invalid})
        with pytest.raises(ValueError, match="Incomplete or duplicated"):
            summarize(prepared, models, tmp_path / "summary.json")
    write_json(folder / "evaluation.json", {**evaluation, "seed": 43})
    with pytest.raises(ValueError, match="identity"):
        summarize(prepared, models, tmp_path / "summary.json")


def test_diagnostic_checkpoint_cannot_load_as_hardware_policy(tmp_path):
    from torch_world.sim2real.experiment import load_checkpoint

    config, task = reference_setup(size=16)
    path = tmp_path / "oracle.pt"
    torch.save({"simulation_diagnostic": True, "oracle": True, "weights": make_oracle(17).state_dict()}, path)
    with pytest.raises(ValueError, match="Checkpoint"):
        load_checkpoint(path, config, task)
