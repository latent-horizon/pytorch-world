import torch

from torch_world.models import WorldModel
from torch_world.planning import CEM


class Integrator(WorldModel):
    def predict(self, z, actions):
        return z + actions


def test_cem_reaches_goal_with_linear_dynamics():
    z, goal = torch.zeros(2, 1, 2), torch.tensor([[1.0, -0.5], [-0.3, 0.6]])
    plan = CEM(horizon=4, action_dim=2, num_samples=200, num_elites=20, iterations=8, bounds=(-0.5, 0.5))
    actions = plan(Integrator(), z, goal, generator=torch.Generator().manual_seed(0))
    assert actions.shape == (2, 4, 2)
    assert torch.allclose(actions.sum(1), goal, atol=0.05)


def test_cem_path_objective_moves_first():
    z, goal = torch.zeros(1, 1, 1), torch.ones(1, 1)
    plan = CEM(horizon=3, action_dim=1, num_samples=500, num_elites=20, iterations=10, bounds=(-1, 1), objective="path")
    actions = plan(Integrator(), z, goal, generator=torch.Generator().manual_seed(0))
    assert actions[0, 0, 0] > 0.8 and (actions.sum() - 1).abs() < 0.1


def test_cem_keeps_past_actions_fixed():
    class Recorder(Integrator):
        def rollout(self, z, actions, **cond):
            self.seen = actions
            return super().rollout(z, actions, **cond)

    model = Recorder()
    past = torch.ones(1, 2, 2)
    CEM(horizon=3, action_dim=2, num_samples=10, num_elites=2, iterations=1)(
        model, torch.zeros(1, 3, 2), torch.zeros(1, 2), past
    )
    assert model.seen.shape == (10, 5, 2) and (model.seen[:, :2] == 1).all()


def test_base_rollout_respects_history():
    class LastOnly(WorldModel):
        history = 1

        def predict(self, z, actions):
            assert z.size(1) == 1
            return z + actions

    out = LastOnly().rollout(torch.zeros(1, 1, 1), torch.ones(1, 3, 1))
    assert out.squeeze().tolist() == [0, 1, 2, 3]


def test_receding_horizon_runs_chunks_and_replans():
    from torch_world.planning import RecedingHorizon

    def plan(obs):
        # horizon 3, frameskip 2, action_dim 1: chunk k holds actions (k, k + 0.5)
        steps = torch.tensor([[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]])
        return steps.expand(len(obs), -1, -1)

    policy = RecedingHorizon(plan, action_dim=1, execute=2)
    actions = [policy(torch.zeros(2, 1))[0, 0].item() for _ in range(6)]
    assert actions == [0.0, 0.5, 1.0, 1.5, 0.0, 0.5] and policy.calls == 2
