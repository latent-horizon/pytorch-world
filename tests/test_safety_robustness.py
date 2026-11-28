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
    from torch_world.benchmarks.safety_robustness import candidate_plans, forecast, tasks
    from torch_world.models import PLDM, PLDMEncoder
    from torch_world.safety import FailureMargin

    model = PLDM(PLDMEncoder(32, 8, channels=(4,)), 2, latent_dim=8).eval()
    margin = FailureMargin(8)
    pair = tasks(8, 203, "cpu", boundary=True)
    actions, families = candidate_plans(*pair, 204, {"PLDM": (model, margin), "second": (model, margin)})
    assert torch.equal(actions[:, families["PLDM CEM"]], actions[:, families["second CEM"]])
    assert actions[:, families["random"]].shape[1] == 6
    clean = forecast(model, margin, pair[0], actions)
    tinted = forecast(model, margin, pair[0], actions, tinted=True)
    delayed = forecast(model, margin, pair[0], actions, delay=True)
    for horizon in (1, 4, 8):
        labels = clean["imagined", horizon][1]
        assert torch.equal(labels, tinted["imagined", horizon][1])
        assert torch.equal(labels, clean["observed", horizon][1])
        assert not labels[:, 1].any()
    assert clean["imagined", 1][1].any()
    assert not delayed["imagined", 1][1].any()


def test_crossing_pool_corner_is_unsafe_before_clear_endpoint():
    from torch_world.benchmarks.safety_robustness import rooms

    env = rooms(1, "cpu")
    env.reset(torch.tensor([[0.18, 0.42]]))
    env.step(torch.tensor([[1.0, -1.0]]))
    assert env.clearance(env.state).item() > 0
    assert env.step_clearance.item() < 0


@pytest.mark.parametrize("step, blocked", [(6, False), (7, True), (8, True), (14, True), (15, False), (16, False)])
def test_gate_checks_each_execution_of_the_requested_command(step, blocked):
    from torch_world.benchmarks.safety_robustness import rooms, simulator_blocks

    probe = rooms(1, "cpu")
    start = torch.tensor([[0.06, 0.5]])
    command = torch.tensor([[1.0, 0.0]])
    assert simulator_blocks(probe, start, command, command, "delay", step).item() == blocked


def test_each_planner_supplies_its_own_commands_to_the_shared_replay(monkeypatch):
    from torch_world.benchmarks import safety_robustness as study
    from torch_world.models import PLDM, PLDMEncoder
    from torch_world.safety import FailureMargin

    first = PLDM(PLDMEncoder(32, 8, channels=(4,)), 2, latent_dim=8).eval()
    second = PLDM(PLDMEncoder(32, 8, channels=(4,)), 2, latent_dim=8).eval()
    margin = FailureMargin(8)

    def planner(model, z, target, past_actions, generator):
        action = z.new_zeros(len(z), 4, 2)
        action[..., 0] = 1 if model is first else -1
        return action

    monkeypatch.setattr(study, "make_planner", lambda: planner)
    start = torch.tensor([[0.18, 0.5]])
    actions, families = study.candidate_plans(start, start, 204, {"A": (first, margin), "B": (second, margin)})
    assert actions.shape == (1, 10, 8, 2)
    for model in (first, second):
        truth = study.forecast(model, margin, start, actions)["imagined", 1][1]
        assert truth[:, families["A CEM"]].item()
        assert not truth[:, families["B CEM"]].item()


def test_exit_drops_pending_request_and_resumes_immediate_screening():
    from torch_world.benchmarks.safety_robustness import rooms, simulator_blocks

    probe = rooms(1, "cpu")
    start = torch.tensor([[0.18, 0.5]])
    command = torch.tensor([[1.0, 0.0]])
    stopped = torch.zeros_like(command)
    assert not simulator_blocks(probe, start, command, stopped, "delay", 15).item()
    assert simulator_blocks(probe, start, command, stopped, "delay", 16).item()
