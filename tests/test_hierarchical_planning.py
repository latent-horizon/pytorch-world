import pytest
import torch

from tests.models.test_hjepa import hjepa
from tests.models.test_hwm import hwm
from torch_world.planning import CEM
from torch_world.planning.hierarchical import GradientPlanner, HierarchicalPlanner


@pytest.mark.parametrize("build", [hwm, hjepa])
@pytest.mark.parametrize("planner", [CEM, GradientPlanner])
def test_hierarchical_plan_shapes(build, planner):
    model = build().eval()
    macro_dim = model.high.macro_dim
    kwargs = {"num_samples": 16, "iterations": 2, "bounds": (-1, 1)}
    if planner is CEM:
        kwargs["num_elites"] = 4
    hierarchical = HierarchicalPlanner(planner(3, macro_dim, **kwargs), planner(model.stride, 2, **kwargs))
    z, goal = model.encode(torch.randn(2, 1, 3, 8, 8)), model.encode(torch.randn(2, 1, 3, 8, 8))[:, 0]
    actions = hierarchical(model, z, goal, generator=torch.Generator().manual_seed(0))
    assert actions.shape == (2, model.stride, 2) and actions.isfinite().all() and actions.abs().max() <= 1
    assert hierarchical.subgoals.shape == (2, 3, 32)
    assert all(p.grad is None for p in model.parameters())


def test_horizon_shrinks_and_anchors_goal():
    model = hwm().eval()
    high = CEM(3, model.high.macro_dim, num_samples=8, num_elites=2, iterations=1)
    hierarchical = HierarchicalPlanner(high, CEM(model.stride, 2, num_samples=8, num_elites=2, iterations=1), 2)
    z, goal = model.encode(torch.randn(1, 1, 3, 8, 8)), model.encode(torch.randn(1, 1, 3, 8, 8))[:, 0]
    horizons = []
    for _ in range(6):
        hierarchical(model, z, goal)
        horizons.append(hierarchical.subgoals.size(1))
    assert horizons == [3, 3, 2, 2, 1, 1] and high.horizon == 3
    hierarchical.reset()
    assert hierarchical.horizon() == 3


def test_gradient_planner_descends():
    model = hwm().eval()
    z = model.encode(torch.randn(3, 1, 3, 8, 8))
    goal = model.rollout(z, torch.ones(3, 4, 2) * 0.5)[:, -1]
    plan = GradientPlanner(4, 2, num_samples=4, iterations=0)
    start = model.cost(model.rollout(z, plan(model, z, goal, torch.Generator().manual_seed(0))), goal)
    plan.iterations = 40
    end = model.cost(model.rollout(z, plan(model, z, goal, torch.Generator().manual_seed(0))), goal)
    assert (end <= start + 1e-6).all() and (end < start).any()
