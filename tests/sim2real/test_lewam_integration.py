import pytest
import torch

from torch_world.sim2real import (
    ChessTaskConfig,
    Episode,
    LeWAMWindows,
    Observation,
    SO101Config,
    WorldModelPolicy,
    load_checkpoint,
    make_adapter,
    save_checkpoint,
)
from torch_world.sim2real.experiment import heldout_loss, train


def episodes():
    data = []
    for delay, success in ((0, True), (1, False)):
        # A delayed sequence is [0, 0, 1, 2, ...], and action i acts at state i.
        time = (torch.arange(9) - delay).clamp_min(0).float()
        data.append(
            Episode(
                time[:, None, None, None].expand(9, 3, 16, 16) / 10,
                time[:, None].expand(9, 6),
                torch.arange(8)[:, None].expand(8, 4).float() / 10,
                torch.zeros(8, 4),
                {"randomization": {"observation_delay": delay}, "outcome": {"success": success}},
            )
        )
    return data


def test_aligned_windows_goals_and_failed_action_masks():
    dataset = LeWAMWindows(episodes(), 3, 2)
    assert len(dataset) == 13
    for i, (episode, t) in enumerate(dataset.index):
        sample = dataset[i]
        assert sample["pixels"].shape == (5, 3, 16, 16)
        assert sample["actions"].shape == (4, 4)
        assert sample["states"][2, 0] == t
        assert sample["actions"][2, 0] == pytest.approx(t / 10)
        assert sample["states"][3, 0] == t + 1
        assert sample["action_valid"].eq(float(episode == 0)).all()
        torch.testing.assert_close(sample["goal_pixels"], episodes()[episode].pixels[-1])
    assert dataset[0]["states"][:3].eq(0).all()


def test_adapter_training_policy_and_portable_checkpoint(tmp_path, monkeypatch):
    torch.manual_seed(4)
    adapter = make_adapter("lewam", size=16, smoke=True, proprioception=True, chunk_size=2)
    losses = train(adapter, episodes(), updates=2, batch_size=2)
    assert len(losses) == 2 and torch.isfinite(torch.tensor(heldout_loss(adapter, episodes())))
    config, task = SO101Config(size=16), ChessTaskConfig()
    path = tmp_path / "model.pt"
    save_checkpoint(path, adapter, config, task)
    restored = load_checkpoint(path, config, task)
    assert restored.recipe == adapter.recipe
    obs = Observation(torch.rand(3, 16, 16), torch.zeros(5), 0.5, 0)
    policy = WorldModelPolicy(restored, obs)
    seen = []
    sample_actions = restored.model.sample_actions

    def capture_history(z, *args, **kwargs):
        seen.append(z.detach().clone())
        return sample_actions(z, *args, **kwargs)

    monkeypatch.setattr(restored.model, "sample_actions", capture_history)
    for _ in range(4):
        action = policy(obs)
        assert action.shape == (4,) and abs(action[3]) <= 0.15
        with pytest.raises(RuntimeError, match="record_action"):
            policy(obs)
        policy.record_action(action)
    assert seen[0].shape[1] == restored.history
    torch.testing.assert_close(seen[0][:, :1].expand_as(seen[0]), seen[0])
    policy.reset()
    assert not policy.pending and len(policy.observations) == 0
    sample = LeWAMWindows(episodes(), 3, 2)[0]
    batch = {k: v[None].expand(2, *v.shape) for k, v in sample.items()}
    g = torch.get_rng_state()
    first = adapter.loss(**batch)
    torch.set_rng_state(g)
    second = restored.loss(**batch)
    for key in first:
        torch.testing.assert_close(first[key], second[key])


def test_heldout_diagnostics_are_finite_and_do_not_train():
    from torch_world.sim2real.diagnostics import diagnose_lewam

    adapter = make_adapter("lewam", size=16, smoke=True, proprioception=True, chunk_size=2)
    before = {k: v.clone() for k, v in adapter.state_dict().items()}
    report = diagnose_lewam(adapter, episodes(), windows_per_episode=2)
    assert report["windows"] == 4 and report["successful_action_windows"] == 2
    assert report["model_one_step_joint_mae_deg"] == report["persistence_one_step_joint_mae_deg"]
    assert all(torch.isfinite(torch.tensor(v)) for v in report.values() if isinstance(v, (float, int)))
    for key, value in adapter.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_observation_alias_diagnostic_exposes_hidden_action_changes():
    from torch_world.sim2real.investigation import observation_aliases

    # Identical observations but different commands: an exact synthetic alias.
    episode = episodes()[0]
    episode.pixels.zero_()
    episode.proprio.zero_()
    episode.actions.zero_()
    episode.actions[3:, 2] = 1
    report = observation_aliases(episode, history=3, chunk_size=2, limit=2)
    assert len(report["pairs"]) == 2
    for pair in report["pairs"]:
        assert pair["rgb_history_mse"] == pair["max_joint_difference_rad"] == 0
        assert abs(pair["actions"][0][2] - pair["actions"][1][2]) == 1
    episode.actions.zero_()
    assert observation_aliases(episode)["pairs"] == []
