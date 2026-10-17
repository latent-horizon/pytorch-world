from dataclasses import asdict

import numpy as np
import pytest
import torch

from torch_world.sim2real import (
    Episode,
    LeWAMWindows,
    Observation,
    RandomizationConfig,
    SO101Chess,
    WorldModelPolicy,
    load_checkpoint,
    make_adapter,
    reference_setup,
    save_checkpoint,
)
from torch_world.sim2real.curriculum import evaluation_cases, matched_adapter, recovery_schedule
from torch_world.sim2real.experiment import train
from torch_world.sim2real.investigation import trace_episode


def test_paired_initialization_and_disjoint_disturbances():
    short = matched_adapter("h3_clean", 17, 16)
    long = matched_adapter("h16_clean", 17, 16)
    for key, value in short.state_dict().items():
        if key == "model.history_position":
            torch.testing.assert_close(long.state_dict()[key][:, -3:], value, rtol=0, atol=0)
        else:
            torch.testing.assert_close(long.state_dict()[key], value, rtol=0, atol=0)
    assert long.history == 16
    train_seeds = set(range(30000, 30012))
    assert not train_seeds & {c["seed"] for c in evaluation_cases()}
    schedule = recovery_schedule(30000, 3)
    assert schedule == recovery_schedule(30000, 3)
    assert np.max(np.abs(schedule[3][:3])) <= 0.6


def test_interventions_do_not_become_imitation_targets():
    ep = Episode(
        torch.zeros(10, 3, 16, 16),
        torch.zeros(10, 10),
        torch.zeros(9, 4),
        torch.zeros(9, 4),
        {"outcome": {"success": True}, "randomization": {"observation_delay": 1}},
        torch.tensor([1, 1, 0, 1, 1, 1, 1, 1, 1]).float(),
    )
    windows = LeWAMWindows([ep], 16, 3)
    assert len(windows) == 6
    torch.testing.assert_close(windows[0]["action_valid"], torch.tensor([1.0, 1.0, 0.0]))
    torch.testing.assert_close(windows[2]["action_valid"], torch.tensor([0.0, 1.0, 1.0]))
    assert windows[0]["states"].shape == (19, 10)
    ep.metadata["outcome"]["success"] = False
    assert not windows[0]["action_valid"].any()
    ep.action_valid = torch.ones(8)
    with pytest.raises(ValueError, match="action_valid"):
        LeWAMWindows([ep], 3, 3)


def test_commanded_state_policy_checkpoint_and_goal(tmp_path):
    pytest.importorskip("mujoco")
    config, task = reference_setup(size=16)
    with SO101Chess(config, task, randomization=RandomizationConfig(enabled=False)) as env:
        env.reset(seed=10007, source="b2", target="a5")
        obs = env.step([0.2, 0, 0, -0.1])
        assert obs.state(True).shape == (10,) and obs.proprio.shape == (6,)
        np.testing.assert_allclose(obs.commanded[:3], env.controller.target, atol=1e-7)
        before = env.controller.target.copy()
        goal = env.goal()
        np.testing.assert_array_equal(before, env.controller.target)
        np.testing.assert_allclose(goal.commanded[:3], env.controller.pose(env.reference_joints), atol=1e-7)
        assert goal.commanded[3] == pytest.approx(1)
        adapter = make_adapter("lewam", size=16, history=16, smoke=True, proprioception=True, command_state=True)
        policy = WorldModelPolicy(adapter, goal)
        action = policy(obs)
        assert action.shape == (4,) and torch.isfinite(action).all()
        policy.record_action(action)
        path = tmp_path / "model.pt"
        save_checkpoint(path, adapter, config, task)
        loaded = load_checkpoint(path, config, task)
        assert loaded.recipe["command_state"] and loaded.history == 16
        torch.testing.assert_close(adapter.proprio(obs.state(True)), loaded.proprio(obs.state(True)))
        missing = Observation(obs.rgb, obs.joints, obs.opening, obs.timestamp)
        with pytest.raises(ValueError, match="recorded commanded"):
            WorldModelPolicy(adapter, missing)


def test_observed_expert_and_masked_recovery_training():
    pytest.importorskip("mujoco")
    config, task = reference_setup(size=16)
    with SO101Chess(config, task, randomization=RandomizationConfig(enabled=False)) as env:
        ep, result = trace_episode(
            env,
            seed=10007,
            source="b2",
            target="a5",
            expert_transition="observed",
            include_commanded=True,
            perturbations=recovery_schedule(30000, 3),
        )
    assert result["outcome"]["success"]
    assert ep.proprio.shape[-1] == 10 and ep.action_valid[3] == 0
    assert ep.action_valid.sum() == len(ep.actions) - 1
    restored = Episode(**asdict(ep))
    adapter = make_adapter("lewam", size=16, history=16, smoke=True, proprioception=True, command_state=True)
    assert np.isfinite(train(adapter, [restored], updates=2, batch_size=2)).all()


def test_summary_rejects_missing_duplicate_and_mismatched_trials(tmp_path):
    import json

    from torch_world.sim2real.curriculum import summarize, write_json

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    folder = tmp_path / "models/h3_clean-seed17"
    folder.mkdir(parents=True)
    protocol = {
        "training": {},
        "controls": [],
        "variants": {"h3_clean": {}},
        "model_seeds": [17],
        "policy_seeds": [101],
        "noise_scales": [0.0],
        "evaluation_cases": [{"name": "nominal"}, {"name": "disturbance_4"}],
        "dataset_sha256": "test",
        "updates": 2000,
    }
    training = {"variant": "h3_clean", "seed": 17, "dataset_sha256": "test", "updates": 2000}
    trials = [
        {
            "noise_scale": 0.0,
            "case": case["name"],
            "policy_seed": 101,
            "outcome": {"success": True, "lifted": True},
            "failure": None,
            "trace": [0],
        }
        for case in protocol["evaluation_cases"]
    ]
    evaluation = {"variant": "h3_clean", "seed": 17, "trials": trials, "tracking": {}}
    write_json(prepared / "protocol.json", protocol)
    write_json(folder / "training.json", training)
    write_json(folder / "evaluation.json", evaluation)
    (folder / "model.pt").write_bytes(b"test")
    output = tmp_path / "summary.json"
    report = summarize(prepared, tmp_path / "models", output)
    assert report["complete"] and report["next_stage_candidates"] == ["h3_clean"]
    assert report["summary"][0]["success_by_model_seed"] == {"17": 1}
    assert "trace" not in report["runs"][0]["trials"][0]
    assert json.loads(output.read_text())["complete"]
    for bad in (trials[:1], trials + trials[:1], trials[:1] * 2):
        write_json(folder / "evaluation.json", {**evaluation, "trials": bad})
        with pytest.raises(ValueError, match="Incomplete or duplicated"):
            summarize(prepared, tmp_path / "models", output)
    write_json(folder / "evaluation.json", {**evaluation, "seed": 29})
    with pytest.raises(ValueError, match="identity"):
        summarize(prepared, tmp_path / "models", output)
    write_json(folder / "training.json", {**training, "updates": 100})
    with pytest.raises(ValueError, match="budgets"):
        summarize(prepared, tmp_path / "models", output)
