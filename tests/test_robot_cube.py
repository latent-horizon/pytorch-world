import pytest
import torch

pytest.importorskip("mujoco")

from torch_world.envs import RobotCube
from torch_world.evaluation import goal_reaching


@pytest.fixture(scope="module", params=["ur10e", "so101"])
def env(request):
    try:
        return RobotCube(request.param, batch_size=2, size=16)
    except OSError as error:
        pytest.skip(f"cannot download the robot model: {error}")


def test_expert_picks_and_places(env):
    env.reset(generator=torch.Generator().manual_seed(0))
    target = env.sample_targets(generator=torch.Generator().manual_seed(1))
    lifted = False
    for _ in range(40):
        env.step(env.expert(target))
        lifted |= bool((env.state[:, 6] > env.half + 0.01).any())
    goal = torch.cat([env.state[:, :4], target, env.state[:, 6:]], dim=-1)
    assert lifted and env.success(env.state, goal).any()


def test_reset_to_state_and_render(env):
    state = env.reset(generator=torch.Generator().manual_seed(2)) is not None and env.state.clone()
    pixels = env.reset(state)
    assert torch.allclose(env.state, state, atol=2e-3)
    assert pixels.shape == (2, 3, 16, 16) and 0 <= pixels.min() and pixels.max() <= 1
    assert env.render(state, camera="wrist").shape == (2, 3, 16, 16)
    assert env.position(state).shape == (2, 2) and ((env.position(state) >= 0) & (env.position(state) <= 1)).all()


def test_reach_tasks_are_solved_by_the_scripted_policy(env):
    env.task = "reach"
    try:
        start, goal = env.tasks(generator=torch.Generator().manual_seed(3))
        assert torch.allclose(env.state, start, atol=2e-3) and (env.distance(start, goal) > 0).all()
        for _ in range(12):
            env.step(env.scripted(goal))
        assert env.success(env.state, goal).all()
        assert torch.allclose(env.state[:, 4:], start[:, 4:], atol=5e-3)
    finally:
        env.task = "pick"


def test_pick_goals_are_finished_and_the_expert_meets_them(env):
    start, goal = env.tasks(generator=torch.Generator().manual_seed(4))
    assert (goal[:, 6] < env.half + 0.01).all() and (env.distance(start, goal) > env.radius).all()
    results = goal_reaching(env, {"expert": lambda o: env.scripted(goal)}, start, goal, 40)
    summary = results.summary()["expert"]
    assert summary["reached"] == 1.0 and 0 < summary["SPL"] <= 1
    assert results.positions["expert"].shape[-1] == 2
