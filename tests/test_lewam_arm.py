import numpy as np
import pytest
import torch

from torch_world.benchmarks.lewam_arm import ArmConfig, environment, expert, task, windows


def test_windows_align_initial_padding_and_episode_goals():
    c = ArmConfig(history=4, chunk_size=5, steps=30)
    times = torch.arange(31)
    pixels = torch.stack([times, times + 100])[:, :, None, None, None].expand(-1, -1, 3, 8, 8).byte()
    data = {
        "pixels": pixels,
        "actions": torch.stack([times[:-1], times[:-1] + 100])[:, :, None].expand(-1, -1, 2).float(),
        "states": torch.stack([times, times + 100])[:, :, None].expand(-1, -1, 4).float(),
        "goal": torch.tensor([[30.0] * 4, [130.0] * 4]),
        "goal_pixels": pixels[:, -1],
    }
    b = windows(data, c, 256, torch.Generator().manual_seed(5))
    p = ((b["pixels"][:, :, 0, 0, 0] + 1) * 127.5).round()
    targets = b["actions"][:, c.history - 1 :]
    torch.testing.assert_close(targets[:, :, 0], p[:, c.history - 1 : -1])
    torch.testing.assert_close(b["state"][:, 0], p[:, c.history - 1])
    assert torch.equal(b["goal"][:, 0] // 100, p[:, c.history - 1] // 100)
    initial = (b["state"][:, 0] % 100) == 0
    assert initial.any()
    assert (b["actions"][initial, : c.history - 1] == 0).all()
    assert (p[initial, : c.history] == p[initial, c.history - 1 : c.history]).all()


def test_owned_environment_uses_native_success_and_expert():
    pytest.importorskip("mujoco")
    c = ArmConfig()
    with environment(c, observation="state") as env:
        assert env.spec.id == "ObstacleArm-v0"
        start, goal = task(20017)
        env.goal = goal[None]
        env.reset(start[None])
        assert not env.success(env.state, env.goal).item()
        for _ in range(c.steps):
            action = expert(env)
            assert torch.isfinite(action).all() and action.abs().max() <= 1
            env.step(action)
        assert env.success(env.state, env.goal).item()
        # Pose alone must not count as settled success.
        moving = env.goal.clone()
        moving[:, 2:] = 2
        assert not env.success(moving, env.goal).item()
        assert np.allclose(env.scene.tolerance, (0.12, 0.12))


def test_seed_zero_and_invalid_budgets():
    assert ArmConfig(seed=0).seed == 0
    with pytest.raises(ValueError):
        ArmConfig(batch_size=0)
    with pytest.raises(ValueError):
        ArmConfig(steps=5, chunk_size=5)
    with pytest.raises(ValueError, match="reference=True"):
        ArmConfig(frameskip=5)
    with pytest.raises(ValueError, match="history_stride"):
        ArmConfig(reference=True, size=224, history_stride=6, frameskip=5)


def test_archived_protocol_keeps_test_tasks_disjoint_and_comparisons_paired():
    import json
    from dataclasses import asdict
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples/action/reports/lewam-owned-arm.json"
    if not path.exists():
        pytest.skip("the LeWAM report is a local archive, not tracked in git")
    report = json.loads(path.read_text())
    assert report["environment"] == "ObstacleArm-v0"
    assert asdict(ArmConfig(**report["config"])) == asdict(ArmConfig())
    splits = [set(seeds) for seeds in report["episode_seeds"].values()]
    assert sum(map(len, splits)) == len(set.union(*splits))
    test_seeds = report["episode_seeds"]["test"]
    for trials in report["evaluation"].values():
        assert [t["seed"] for t in trials] == test_seeds
        assert not any(t["initial_success"] for t in trials)
    assert all(t["success"] and t["final_success"] for t in report["evaluation"]["expert"])


def test_reference_windows_use_strided_history_consecutive_actions_and_tail_mask():
    c = ArmConfig(reference=True, size=224, history=2, history_stride=5, chunk_size=10, frameskip=5, steps=30)
    times = torch.arange(31)
    pixels = times[None, :, None, None, None].expand(1, -1, 3, 2, 2).byte()
    data = {
        "pixels": pixels,
        "actions": times[None, :-1, None].expand(1, -1, 2).float(),
        "states": times[None, :, None].expand(1, -1, 4).float(),
        "goal": torch.ones(1, 4) * 30,
        "goal_pixels": pixels[:, -1],
    }
    b = windows(data, c, 512, torch.Generator().manual_seed(2))
    current = b["state"][:, 0]

    def decode(x):
        return ((x[..., 0, 0, 0] * 0.229 + 0.485) * 255).round()

    h = decode(b["history_pixels"])
    torch.testing.assert_close(h[:, -1], current)
    torch.testing.assert_close(decode(b["future_pixels"])[:, 0], current + 5)
    torch.testing.assert_close(b["actions"][:, 0, 0], current)
    pad = current < 5
    assert pad.any() and (~b["action_valid"]).any()
    torch.testing.assert_close(h[:, 0], torch.where(pad, current, current - 5))
    assert torch.equal(b["history_pad"][:, 0], pad)
    assert torch.equal(b["action_valid"], current[:, None] + torch.arange(10) < 30)


def test_evaluation_rejects_invalid_execution_interval_before_creating_an_env(tmp_path):
    from torch_world.benchmarks.lewam_arm import evaluate

    with pytest.raises(ValueError, match="execution_interval"):
        evaluate(None, None, ArmConfig(), [0], tmp_path, "cpu", execution_interval=0)
    with pytest.raises(ValueError, match="controller"):
        evaluate(None, None, ArmConfig(), [0], tmp_path, "cpu", controllers=["typo"])


def test_cadence_evaluation_rejects_mismatched_checkpoint(tmp_path):
    import json

    from torch_world.benchmarks.lewam_arm import evaluate_checkpoint

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"not the reported checkpoint")
    (tmp_path / "results.json").write_text(json.dumps({"checkpoint_sha256": "incorrect"}))
    output = tmp_path / "cadence"
    with pytest.raises(ValueError, match="Checkpoint does not match"):
        evaluate_checkpoint(checkpoint, output, device="cpu")
    assert not output.exists()


def test_phase_diagnostic_includes_launch_and_handles_short_episodes():
    from torch_world.benchmarks.lewam_arm import diagnose_phases

    config = ArmConfig(
        reference=True, size=224, history=2, frameskip=5, history_stride=5, chunk_size=10, steps=30, mean_samples=1
    )
    pixels = torch.arange(31).byte()[None, :, None, None, None].expand(2, -1, 3, 8, 8)
    data = {
        "pixels": pixels,
        "goal_pixels": pixels[:, -1],
        "actions": torch.arange(30)[None, :, None].expand(2, -1, 2).float() / 30,
    }
    masks = []

    class Policy:
        chunk_size = 10

        def encode(self, images):
            return images.mean((-1, -2))

        def sample_actions(self, z, goal, *, history_pad=None, **kwargs):
            masks.append(history_pad.clone())
            return z.new_zeros(len(z), self.chunk_size, 2)

    rows = diagnose_phases(Policy(), data, config, "cpu")
    assert [r["step"] for r in rows] == [0, 5, 20]
    assert all(r["episodes"] == 2 for r in rows)
    assert masks[0][:, 0].all() and not masks[0][:, 1].any()
    assert not masks[1].any()
    assert rows[0]["action_mse"] == 0
    assert rows[-1]["expert_mean"] == pytest.approx([20 / 30, 20 / 30])


def test_launch_sampling_and_explicit_validation_windows():
    from dataclasses import replace

    c = ArmConfig(
        reference=True,
        size=224,
        history=2,
        history_stride=5,
        chunk_size=10,
        frameskip=5,
        steps=30,
        launch_fraction=1,
        phase_validation=True,
    )
    pixels = torch.zeros(3, 31, 3, 2, 2, dtype=torch.uint8)
    times = torch.arange(31).float()
    data = {
        "pixels": pixels,
        "goal_pixels": pixels[:, -1],
        "actions": times[None, :-1, None].expand(3, -1, 2),
        "states": times[None, :, None].expand(3, -1, 4),
        "goal": torch.zeros(3, 4),
    }
    batch = windows(data, c, 128, torch.Generator().manual_seed(11))
    assert batch["state"][:, 0].min() == 0 and batch["state"][:, 0].max() == 4
    explicit = windows(
        data,
        replace(c, launch_fraction=0),
        3,
        torch.Generator().manual_seed(11),
        episode_ids=torch.arange(3),
        start_steps=torch.tensor([0, 1, 20]),
    )
    torch.testing.assert_close(explicit["state"][:, 0], torch.tensor([0.0, 1.0, 20.0]))
    assert explicit["history_pad"][:, 0].tolist() == [True, True, False]


def test_adjacent_history_exposes_motion_after_first_step():
    c = ArmConfig(reference=True, size=224, history=2, frameskip=5, chunk_size=10, history_stride=1)
    times = torch.arange(101).float()
    pixels = times[None, :, None, None, None].expand(1, -1, 3, 2, 2).byte()
    data = {
        "pixels": pixels,
        "actions": times[None, :-1, None].expand(1, -1, 2),
        "states": times[None, :, None].expand(1, -1, 4),
        "goal": torch.zeros(1, 4),
        "goal_pixels": pixels[:, -1],
    }
    batch = windows(
        data, c, 3, torch.Generator(), episode_ids=torch.zeros(3, dtype=torch.long), start_steps=torch.tensor([0, 1, 4])
    )
    assert batch["history_pad"].tolist() == [[True, False], [False, False], [False, False]]
    decoded = ((batch["history_pixels"][:, :, 0, 0, 0] * 0.229 + 0.485) * 255).round()
    torch.testing.assert_close(decoded, torch.tensor([[0.0, 0.0], [0.0, 1.0], [3.0, 4.0]]))
    # Prediction spacing stays five steps: it must not set observation cadence.
    future = ((batch["future_pixels"][:, :, 0, 0, 0] * 0.229 + 0.485) * 255).round()
    torch.testing.assert_close(future[:, 0], torch.tensor([5.0, 6.0, 9.0]))


def test_trace_records_executed_actions_and_native_states(tmp_path):
    import json

    from torch_world.benchmarks.lewam_arm import evaluate

    c = ArmConfig(execution_interval=1)
    result = evaluate(None, None, c, [20017], tmp_path, "cpu", controllers=["expert"])
    trace = json.loads((tmp_path / "trace-expert.json").read_text())
    assert len(trace["states"]) == c.steps + 1
    assert len(trace["actions"]) == c.steps
    assert trace["actions"] == trace["expert_actions"]
    assert trace["states"][0][0] == task(20017)[0].tolist()
    assert result["expert"][0]["final_success"]
    with pytest.raises(ValueError, match="execution_interval"):
        ArmConfig(execution_interval=11)


def test_replay_audit_detects_corrupt_labels():
    from torch_world.benchmarks.lewam_arm import collect
    from torch_world.benchmarks.lewam_arm_diagnostics import audit_demonstrations

    c = ArmConfig(size=32)
    data = collect(c, [c.seed])
    report = audit_demonstrations(data, c)
    row = report["replay"][0]
    assert row["max_state_error"] == row["max_pixel_error_uint8"] == row["goal_render_state_error"] == 0
    assert row["final_success"]
    data["states"][0, 3, 0] += 0.5
    corrupted = audit_demonstrations(data, c)
    assert corrupted["replay"][0]["max_state_error"] == pytest.approx(0.5)


def test_gentler_expert_preserves_native_success():
    c = ArmConfig(expert_gain=2)
    seeds = range(17, 41)
    starts, goals = map(torch.stack, zip(*(task(seed) for seed in seeds)))
    with environment(c, batch_size=len(seeds), observation="state") as env:
        env.goal = goals
        env.reset(starts)
        gentle = expert(env, gain=c.expert_gain)
        aggressive = expert(env)
        assert gentle[:, 1].abs().mean() < aggressive[:, 1].abs().mean()
        for _ in range(c.steps):
            env.step(expert(env, gain=c.expert_gain))
        assert env.success(env.state, env.goal).all()


def test_visual_execution_cadence_uses_config_and_records_cached_chunk(tmp_path):
    import json

    from torch_world.benchmarks.lewam_arm import evaluate

    c = ArmConfig(size=32, steps=12, execution_interval=2)

    class Policy:
        chunk_size = c.chunk_size
        calls = 0

        def encode(self, pixels):
            return pixels.mean((-1, -2))

        def sample_actions(self, z, goal, **kwargs):
            self.calls += 1
            return torch.arange(self.chunk_size)[None, :, None].expand(len(z), -1, 2).float() / 10

    policy = Policy()
    evaluate(policy, None, c, [20017], tmp_path, "cpu", controllers=["lewam_single"])
    trace = json.loads((tmp_path / "trace-lewam_single.json").read_text())
    assert policy.calls == 6
    assert torch.tensor(trace["actions"])[:, 0, 0].tolist() == pytest.approx([0, 0.1] * 6)


def test_recovery_collection_changes_starts_only_and_replays_exactly():
    from torch_world.benchmarks.lewam_arm import collect, demonstration_task
    from torch_world.benchmarks.lewam_arm_diagnostics import audit_demonstrations

    c = ArmConfig(size=32, expert_gain=2, recovery_fraction=1)
    start, goal = demonstration_task(c.seed, c)
    original_start, original_goal = task(c.seed)
    assert torch.equal(goal, original_goal)
    assert not torch.equal(start, original_start)
    assert ((start[:2] - goal[:2]).abs() <= 0.35).all() and (start[2:] == 0).all()
    assert torch.equal(demonstration_task(c.seed, ArmConfig())[0], original_start)
    data = collect(c, [c.seed])
    replay = audit_demonstrations(data, c)["replay"][0]
    assert replay["max_state_error"] == replay["max_pixel_error_uint8"] == 0
    assert replay["final_success"]


@pytest.mark.parametrize("change", [{"expert_gain": 2}, {"recovery_fraction": 0.5}])
def test_cached_demonstrations_require_same_collection_protocol(tmp_path, change):
    import json
    from dataclasses import asdict

    from torch_world.benchmarks.lewam_arm import run_arm

    data_path = tmp_path / "episodes.pt"
    (tmp_path / "config.json").write_text(json.dumps(asdict(ArmConfig())))
    with pytest.raises(ValueError, match="Cached demonstrations disagree"):
        run_arm(tmp_path / "output", ArmConfig(**change), data_path=data_path, device="cpu")


def test_recovery_curriculum_keeps_fresh_evaluation_and_data_protocols(tmp_path, monkeypatch):
    import torch_world.benchmarks.lewam_arm as arm

    calls = []

    def record(output, config, **kwargs):
        calls.append((output, config, kwargs))
        return {"stage": output.name}

    monkeypatch.setattr(arm, "run_arm", record)
    c = ArmConfig(
        reference=True,
        size=224,
        history=2,
        chunk_size=10,
        frameskip=5,
        history_stride=1,
        execution_interval=1,
        expert_gain=2,
        recovery_fraction=0.5,
        test_seed_offset=30000,
        updates=1000,
    )
    result = arm.run_recovery_curriculum(tmp_path / "study", c)
    assert result == {"stage": "recovery"}
    assert [call[1].updates for call in calls] == [2000, 500, 1000, 1000]
    assert [call[1].test_seed_offset for call in calls] == [20000, 20000, 20000, 30000]
    assert [call[1].expert_gain for call in calls] == [10, 10, 2, 2]
    assert [call[1].recovery_fraction for call in calls] == [0, 0, 0, 0.5]
    assert calls[1][2]["data_path"] == tmp_path / "study/initial/episodes.pt"
    assert all("data_path" not in calls[index][2] for index in (0, 2, 3))
    assert calls[-1][2]["initial_checkpoint"] == tmp_path / "study/smooth/model.pt"


def test_archived_matched_tasks_are_disjoint_and_controllers_are_paired():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples/action/reports/lewam-matched.json"
    if not path.exists():
        pytest.skip("the LeWAM report is a local archive, not tracked in git")
    report = json.loads(path.read_text())
    assert report["complete"]
    config = ArmConfig(**report["training"]["config"])
    training_seeds = set(range(config.seed, config.seed + config.train_episodes))
    validation_seeds = set(range(config.seed + 10000, config.seed + 10000 + config.validation_episodes))
    for split, groups in report["groups"].items():
        for models in groups.values():
            paired = [(t["seed"], t["start_step"]) for t in models["local"]["policy"]]
            assert len(paired) == report["cases_per_seed"]
            seeds = {seed for seed, start in paired}
            assert not seeds & validation_seeds
            if split == "training_trajectories":
                assert seeds <= training_seeds
            else:
                assert not seeds & training_seeds
            for controllers in models.values():
                for trials in controllers.values():
                    assert [(t["seed"], t["start_step"]) for t in trials] == paired
                assert all(t["success"] and t["final_success"] for t in controllers["expert"])
