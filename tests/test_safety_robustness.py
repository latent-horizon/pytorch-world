import pytest
import torch

from torch_world.benchmarks.safety_robustness import episode_metrics, forecast_metrics


def test_early_violation_disqualifies_success_and_preserves_peak():
    reached = torch.tensor([[False, True], [True, False]])
    clearance = torch.tensor([[-0.3, 0.1], [0.2, 0.1]])
    result = episode_metrics(reached, clearance, torch.zeros_like(reached))
    assert result["safe success"] == 0.5
    assert result["violation"] == 0.5
    assert result["tail penetration"] == pytest.approx(0.3)


def test_perfect_and_reject_all_forecasts():
    hazard = torch.tensor([[True, False], [False, True]])
    exact = forecast_metrics(hazard.float(), hazard, 0.5)
    assert exact["unsafe acceptance"] == exact["avoidable violation"] == exact["Brier"] == 0
    assert exact["safe acceptance"] == exact["average precision"] == 1
    reject = forecast_metrics(torch.ones_like(hazard).float(), hazard, 0.5)
    assert reject["unsafe acceptance"] == reject["safe acceptance"] == 0
    assert reject["average precision"] == 0.5
    assert reject["accepted risk"] is None


def test_forecast_keeps_undefined_rates_distinct_from_zero():
    events = torch.zeros(2, 3, dtype=torch.bool)
    result = forecast_metrics(events.float(), events, 0.1)
    assert result["unsafe acceptance"] is None
    assert result["average precision"] is None
    assert result["accepted risk"] == 0


def test_delayed_command_uses_queue_only_during_intervention():
    from torch_world.benchmarks.safety_robustness import applied_command

    command, previous = torch.ones(1, 2), torch.zeros(1, 2)
    for step in (7, 16):
        assert torch.equal(applied_command(command, previous, "tint + delay", step), command)
    for step in (8, 15):
        assert torch.equal(applied_command(command, previous, "tint + delay", step), previous)
        assert torch.equal(applied_command(command, previous, "gain 0.5", step), command / 2)


def test_geometry_controller_reaches_held_out_goals_safely():
    from torch_world.benchmarks.safety_robustness import state_controller, tasks

    start, goal = tasks(32, 203, "cpu")
    assert state_controller(start, goal, "clean")["safe success"] == 1


def test_arm_replay_restores_matched_starts_and_measures_clearance():
    pytest.importorskip("mujoco")
    from torch_world.benchmarks.arm_safety import arm_replay

    first = arm_replay(4, 101, 8, 4)
    second = arm_replay(4, 101, 8, 4)
    assert torch.equal(first["clearance"], second["clearance"])
    starts = first["states"][:, 0].reshape(4, 4, -1)
    assert torch.equal(starts, starts[:, :1].expand_as(starts))
    assert (first["clearance"][:, 0] > 0.01).all()
    assert (first["clearance"][:, 1:] < 0.01).any()


def test_sampled_event_probability_counts_trajectories_once():
    from torch_world.benchmarks.safety_robustness import sample_hazard_probability

    clearance = torch.tensor([[[[1, -1, -1], [1, 1, 1], [-1, 1, -1]]]])
    assert sample_hazard_probability(clearance, 1).item() == pytest.approx(1 / 3)
    assert sample_hazard_probability(clearance, 3).item() == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        sample_hazard_probability(clearance, 4)


def test_counterfactual_labels_ignore_tint_and_obey_action_delay():
    from torch_world.benchmarks.safety_robustness import forecast, tasks
    from torch_world.models import PLDM, PLDMEncoder
    from torch_world.safety import FailureMargin

    model = PLDM(PLDMEncoder(32, 8, channels=(4,)), 2, latent_dim=8).eval()
    margin = FailureMargin(8)
    pair = tasks(8, 203, "cpu", boundary=True)
    clean = forecast(model, margin, *pair, seed=204)
    tinted = forecast(model, margin, *pair, seed=204, tinted=True)
    delayed = forecast(model, margin, *pair, seed=204, delay=True)
    for horizon in (1, 4, 8):
        labels = clean["imagined", horizon][1]
        assert torch.equal(labels, tinted["imagined", horizon][1])
        assert torch.equal(labels, clean["observed", horizon][1])
        assert not labels[:, 1].any()
    assert clean["imagined", 1][1].any()
    assert not delayed["imagined", 1][1].any()
