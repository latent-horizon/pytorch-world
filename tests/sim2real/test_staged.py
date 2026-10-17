import pytest
import torch

from torch_world.sim2real import Observation, make_adapter, reference_setup, simulation_gate
from torch_world.sim2real.staged import GoalStage, StagedWorldModelPolicy, scripted_goal_stages


class ZeroPlanner:
    def __init__(self):
        self.goals = []

    def __call__(self, model, latent, goal, **kwargs):
        self.goals.append(goal.clone())
        return latent.new_zeros(1, 1, 4)


def observation(joint=0.0, opening=1.0, timestamp=0.0):
    return Observation(torch.zeros(3, 16, 16), torch.full((5,), joint), opening, timestamp)


def test_stages_use_only_observations_preserve_history_and_reset():
    adapter = make_adapter("lewm", size=16, smoke=True, proprioception=True)
    stages = [GoalStage("approach", observation()), GoalStage("grasp", observation(0.1, 0.3))]
    planner = ZeroPlanner()
    policy = StagedWorldModelPolicy(adapter, stages, planner=planner, dwell_steps=2)
    policy(observation())
    assert policy.index == 0
    with pytest.raises(RuntimeError, match="Record"):
        policy(observation())
    policy.record_action(torch.zeros(4))
    policy(observation(timestamp=0.1))
    assert policy.index == 1 and len(policy.policy.observations) == 2
    assert policy.transitions == [{"stage": "approach", "timestamp": 0.1}]
    assert not torch.allclose(planner.goals[0], planner.goals[1])
    policy.record_action(torch.zeros(4))
    policy.reset()
    assert policy.index == 0 and not policy.transitions and not policy.policy.actions
    assert policy.goal is stages[0].observation
    assert policy.check_latency(observation(), 100, repeats=1)
    assert policy.index == 0 and not policy.policy.pending and not policy.policy.actions


def test_goal_contracts_cannot_be_mixed():
    adapter = make_adapter("lewm", size=16, smoke=True)
    first, second = observation(), observation()
    first.contract, second.contract = "calibration-a", "calibration-b"
    with pytest.raises(ValueError, match="contract"):
        StagedWorldModelPolicy(adapter, [GoalStage("approach", first), GoalStage("grasp", second)])


def test_gripper_and_each_joint_must_reach_stage_before_advancing():
    adapter = make_adapter("lewm", size=16, smoke=True)
    stages = [GoalStage("grasp", observation(0.0, 0.2)), GoalStage("lift", observation(0.2, 0.2))]
    policy = StagedWorldModelPolicy(adapter, stages, planner=ZeroPlanner(), dwell_steps=1)
    for obs in (observation(0.0, 1.0), observation(0.2, 0.2)):
        policy(obs)
        assert policy.index == 0
        policy.record_action(torch.zeros(4))
    policy(observation(0.0, 0.2))
    assert policy.index == 1


def test_scripted_assistance_cannot_pass_progression_gate():
    result = {
        "smoke": False,
        "goal_protocol": "scripted_keyframes",
        "config": {"control_dt": 0.1},
        "evaluation": {
            d: {"learned": {"episodes": [{"success": True, "max_planning_seconds": 0.01, "failure": None}] * 24}}
            for d in ("nominal", "randomized")
        },
    }
    gate = simulation_gate(result)
    assert not gate["passed"] and not gate["physical_trial_enabled"]
    assert any("assistance" in r for r in gate["reasons"])


def test_successful_scripted_reference_provides_all_nine_stages():
    pytest.importorskip("mujoco")
    from torch_world.sim2real import RandomizationConfig, SO101Chess

    config, task = reference_setup(size=16)
    with SO101Chess(config, task, randomization=RandomizationConfig(enabled=False)) as env:
        stages, metadata = scripted_goal_stages(env, seed=10007)
        assert metadata["success"] and env.info()["success"]
        assert [s.name for s in stages] == [
            "approach",
            "grasp",
            "lift",
            "transfer",
            "lower",
            "release",
            "clear",
            "withdraw",
            "return",
        ]
        assert all(s.observation.rgb.shape == (3, 16, 16) for s in stages)


def test_failed_reference_is_not_converted_to_a_goal_sequence():
    class FailedScene:
        task = type("Task", (), {"max_steps": 3})()
        source, target_square, phase, steps = "a1", "b1", 0, 0

        def __init__(self):
            self.episode_parameters = {}

        def reset(self, **kwargs):
            return observation()

        def expert(self):
            return torch.zeros(4)

        def step(self, action):
            raise ValueError("unreachable")

        def info(self):
            return {"success": False}

    stages, metadata = scripted_goal_stages(FailedScene(), seed=7)
    assert not stages and not metadata["success"] and metadata["failure"] == "unreachable"
