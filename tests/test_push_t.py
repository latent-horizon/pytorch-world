import math

import pytest
import torch

pytest.importorskip("mujoco")

from torch_world.envs import PushT
from torch_world.envs.push_t import PUSHER
from torch_world.evaluation import goal_reaching, random_policy


def test_pusher_moves_and_pushes_the_block():
    env = PushT(batch_size=1, size=16)
    env.reset(torch.tensor([[-0.1, -0.03, 0.0, 0.0, 0.0]], dtype=torch.float64))
    for _ in range(12):
        env.step(torch.tensor([[1.0, 0.0]]))
    assert env.state[0, 0] > 0 and env.state[0, 2] > 0.05


def test_tasks_success_and_ghost():
    env = PushT(batch_size=4, size=16)
    start, goal = env.tasks(steps=10, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(env.state, start, atol=1e-4) and not env.success(start, goal).any()
    turned = goal + torch.tensor([0, 0, 0, 0, 2 * math.pi, 0, 0])
    assert env.success(goal, goal).all() and env.angle(turned, goal).max() < 1e-4
    results = goal_reaching(
        env, {"random": random_policy(env)}, start, goal, steps=3, record=lambda e: e.render(goal=goal)
    )
    assert results.frames["random"].shape == (4, 4, 3, 16, 16) and results.frames["random"].dtype == torch.uint8
    assert not torch.equal(env.render(goal=goal), env.render())


def test_coverage_of_the_goal_region():
    env = PushT(batch_size=2, size=16)
    goal = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.3], [0.0, 0.0, 0.1, -0.05, 2.0]])
    assert torch.allclose(env.coverage(goal, goal), torch.ones(2))
    flipped = goal + torch.tensor([0, 0, 0, 0, math.pi])  # turned about the middle of the bar's top edge
    assert env.coverage(flipped, goal).max() == 0
    shifted = goal + torch.tensor([0, 0, 0.5, 0, 0])
    assert env.coverage(torch.stack([goal, shifted], 1), goal).tolist() == [[1.0, 0.0], [1.0, 0.0]]


def test_actions_from_a_recorded_state_replay_the_episode():
    env = PushT(batch_size=8, size=16)
    data = env.collect(30, generator=torch.Generator().manual_seed(0))
    env.reset(data["states"][:, 10])
    for t in range(10, 20):
        env.step(data["actions"][:, t])
    assert (env.state[:, 2:4] - data["states"][:, 20, 2:4]).norm(dim=-1).median() < 2e-3


def test_reference_units_and_success():
    reference = torch.tensor([[423.0, 182.0, 192.7, 335.2, 2.95, 52.4, -93.5]], dtype=torch.float64)
    state = PushT.from_reference(reference)
    assert torch.allclose(PushT.to_reference(state), reference)
    assert torch.allclose(state[0, :2], torch.tensor([0.167, 0.074], dtype=torch.float64))
    env = PushT(batch_size=1, size=16)
    goal = torch.zeros(1, 7)
    near = torch.tensor([[0.012, 0.0, 0.0, 0.012, 0.3, 0.0, 0.0]])  # 17 mm in the stacked positions, 17 degrees
    assert env.success(near, goal).all()
    assert not env.success(near * torch.tensor([1.0, 1, 1, 1.5, 1, 1, 1]), goal).any()
    assert not env.success(near * torch.tensor([1.0, 1, 1, 1, 1.25, 1, 1]), goal).any()


def test_the_block_moves_with_the_pusher_and_stops_with_it():
    """A push through the T's center of gravity moves it straight with the pusher; the block stops with it."""
    env = PushT(batch_size=1, size=16)
    env.reset(torch.tensor([[0.0, 0.1, 0.0, 0.0, 0.0]], dtype=torch.float64))
    for _ in range(8):
        env.step(torch.tensor([[0.0, -0.5]]))
    moved = env.state[0, 3].item()
    assert moved < -0.05 and abs(env.state[0, 2]) < 1e-3 and abs(env.state[0, 4]) < 1e-3
    assert abs(env.state[0, 1] - moved - PUSHER) < 0.005  # the pusher rests on the bar's top edge
    for _ in range(5):
        env.step(torch.tensor([[0.0, 1.0]]))
    assert abs(env.state[0, 3] - moved) < 1e-3
