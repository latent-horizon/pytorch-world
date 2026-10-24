import torch
from torch import nn

from torch_world.models import WorldModel
from torch_world.planning import GRASP


class Linear(WorldModel):
    """``z' = z + B a`` with a learnable ``B``: the plan reaches ``goal`` when ``B (sum_t a_t) = goal - z``."""

    def __init__(self, dim=3, action_dim=2):
        super().__init__()
        self.B = nn.Linear(action_dim, dim, bias=False)
        with torch.no_grad():
            self.B.weight.copy_(torch.randn(dim, action_dim, generator=torch.Generator().manual_seed(0)))

    def predict(self, z, actions):
        return z + self.B(actions)


def problem(batch=3, dim=3, action_dim=2):
    model = Linear(dim, action_dim)
    g = torch.Generator().manual_seed(1)
    z = torch.randn(batch, 1, dim, generator=g)
    with torch.no_grad():
        goal = z[:, 0] + model.B(torch.rand(batch, action_dim, generator=g) * 2 - 1)  # reachable in one step
    return model, z, goal


def final_cost(model, z, goal, actions):
    with torch.no_grad():
        return model.cost(model.rollout(z, actions), goal)


def test_shape_and_bounds():
    model, z, goal = problem()
    goal = goal * 50  # out of reach: the bounds bind
    plan = GRASP(horizon=4, action_dim=2, iterations=60, sync_every=20, lr_actions=0.1, bounds=(-0.3, 0.3))
    actions = plan(model, z, goal)
    assert actions.shape == (3, 4, 2) and not actions.requires_grad
    assert actions.abs().max() <= 0.3 + 1e-6


def test_lowers_the_cost():
    model, z, goal = problem()
    plan = GRASP(horizon=5, action_dim=2, iterations=300, lr_actions=0.05, sync_every=50, sync_steps=10, sync_lr=0.05)
    actions = plan(model, z, goal, generator=torch.Generator().manual_seed(0))
    before = final_cost(model, z, goal, torch.zeros_like(actions))
    after = final_cost(model, z, goal, actions)
    assert (after < 0.1 * before).all()


def test_relaxed_phase_alone_lowers_cost():
    model, z, goal = problem()
    plan = GRASP(horizon=5, action_dim=2, iterations=300, lr_actions=0.05, noise=0.0, sync_every=0)
    actions = plan(model, z, goal)
    assert (final_cost(model, z, goal, actions) < 0.1 * final_cost(model, z, goal, torch.zeros_like(actions))).all()


def test_model_params_get_no_grads_and_works_under_no_grad():
    model, z, goal = problem()
    with torch.no_grad():
        actions = GRASP(horizon=3, action_dim=2, iterations=20, sync_every=10)(model, z, goal)
    assert torch.isfinite(actions).all() and actions.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())


def test_batched_goals_are_planned_independently():
    model, z, goal = problem(batch=4)
    plan = GRASP(horizon=4, action_dim=2, iterations=200, lr_actions=0.05, noise=0.0, sync_every=50)
    batched = plan(model, z, goal)
    alone = torch.cat([plan(model, z[i : i + 1], goal[i : i + 1]) for i in range(4)])
    assert torch.allclose(batched, alone, atol=1e-4)
    assert (final_cost(model, z, goal, batched) < 0.1 * final_cost(model, z, goal, torch.zeros_like(batched))).all()


def test_context_past_actions_and_horizon_one():
    class Recorder(Linear):
        def rollout(self, z, actions, **cond):
            if z.size(1) > 1:
                self.seen = actions
            return super().rollout(z, actions, **cond)

    _, z, goal = problem(batch=2)
    model, past = Recorder(), torch.ones(2, 2, 2)
    context = torch.cat([z - 1, z - 0.5, z], dim=1)
    actions = GRASP(horizon=1, action_dim=2, iterations=10, sync_every=5)(model, context, goal, past)
    assert actions.shape == (2, 1, 2) and model.seen.shape == (2, 3, 2) and (model.seen[:, :2] == 1).all()
