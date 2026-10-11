import math

import pytest
import torch

pytest.importorskip("mujoco")

from torch_world.envs import PushT
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
    assert (
        env.success(goal, goal).all() and env.angle(goal + torch.tensor([0, 0, 0, 0, 2 * math.pi]), goal).max() < 1e-4
    )
    results = goal_reaching(
        env, {"random": random_policy(env)}, start, goal, steps=3, record=lambda e: e.render(goal=goal)
    )
    assert results.frames["random"].shape == (4, 4, 3, 16, 16) and results.frames["random"].dtype == torch.uint8
    assert not torch.equal(env.render(goal=goal), env.render())


def test_coverage_of_the_goal_region():
    env = PushT(batch_size=2, size=16)
    goal = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.3], [0.0, 0.0, 0.1, -0.05, 2.0]])
    assert torch.allclose(env.coverage(goal, goal), torch.ones(2))
    flipped = goal + torch.tensor([0, 0, 0, 0, math.pi])
    assert torch.allclose(env.coverage(flipped, goal), torch.tensor(4 / 7), atol=1e-6)
    shifted = goal + torch.tensor([0, 0, 0.5, 0, 0])
    assert env.coverage(torch.stack([goal, shifted], 1), goal).tolist() == [[1.0, 0.0], [1.0, 0.0]]
