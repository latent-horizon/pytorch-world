import pytest
import torch

from torch_world.models import TDMPC2
from torch_world.models.tdmpc import pixel_encoder, state_encoder
from torch_world.planning import MPPI


def tdmpc2(**kwargs):
    return TDMPC2(state_encoder(6, latent_dim=32, hidden_dim=16), action_dim=2, latent_dim=32, mlp_dim=32, **kwargs)


def batch(B=8, H=3):
    return torch.randn(B, H + 1, 6), torch.rand(B, H, 2) * 2 - 1, torch.randn(B, H)


def test_losses_and_shapes():
    model = tdmpc2()
    obs, actions, rewards = batch()
    out = model(obs, actions, rewards)
    world = 20 * out["consistency_loss"] + 0.1 * out["reward_loss"] + 0.1 * out["value_loss"]
    assert torch.allclose(out["loss"], world + out["pi_loss"])
    assert all(v.isfinite() for v in out.values())
    z = model.encode(obs)
    assert z.shape == (8, 4, 32) and torch.allclose(z.unflatten(-1, (-1, 8)).sum(-1), torch.ones(8, 4, 4))
    assert model.predict(z, torch.zeros(8, 4, 2)).shape == (8, 4, 32)
    assert model.q(z, torch.zeros(8, 4, 2)).shape == (5, 8, 4, 101)
    assert model.q(z, torch.zeros(8, 4, 2), "min").shape == (8, 4)
    assert model.rollout(z[:, :1], actions).shape == (8, 4, 32)


def test_policy_loss_only_reaches_the_policy():
    model = tdmpc2()
    model(*batch())["pi_loss"].backward()
    assert all(p.grad is not None for p in model.policy.parameters())
    assert all(p.grad is None for p in [*model.encoder.parameters(), *model.qs.parameters()])


def test_world_loss_does_not_reach_the_policy():
    model = tdmpc2()
    out = model(*batch())
    (20 * out["consistency_loss"] + 0.1 * out["reward_loss"] + 0.1 * out["value_loss"]).backward()
    assert all(p.grad is None for p in model.policy.parameters())
    assert all(p.grad is not None for p in model.encoder.parameters())


def test_two_hot_round_trip():
    model = tdmpc2()
    x = torch.tensor([-50.0, -1.3, 0.0, 0.7, 12.0])
    target = model.two_hot(x)
    assert torch.allclose(target.sum(-1), torch.ones(5)) and (target > 0).sum(-1).le(2).all()
    assert torch.allclose(model.two_hot_inv(target.clamp_min(1e-12).log()), x, rtol=1e-4, atol=1e-4)


def test_target_and_scale_updates():
    model = tdmpc2().train()
    assert not model.target_qs.training
    model(*batch())
    assert model.q_scale >= 1
    scale = model.q_scale.clone()
    model.eval()(*batch())
    assert torch.equal(model.q_scale, scale)
    for p in model.qs.parameters():
        p.data.add_(1.0)
    before = [p.clone() for p in model.target_qs.parameters()]
    model.update_target(0.99)
    for b, t, q in zip(before, model.target_qs.parameters(), model.qs.parameters()):
        assert torch.allclose(t, 0.99 * b + 0.01 * q)


def test_param_groups_cover_every_parameter():
    model = tdmpc2(episodic=True)
    groups = model.param_groups(1e-3)
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters())) - len(
        list(model.target_qs.parameters())
    )
    assert groups[0]["lr"] == pytest.approx(3e-4) and groups[2]["eps"] == 1e-5


def test_episodic_termination():
    model = tdmpc2(episodic=True)
    obs, actions, rewards = batch()
    out = model(obs, actions, rewards, torch.zeros(8, 3))
    assert "termination_loss" in out and out["termination_loss"].isfinite()
    assert model.estimate_value(model.encode(obs)[:, 0], actions).shape == (8,)


def test_pixel_encoder_latent():
    z = pixel_encoder()(torch.rand(2, 3, 64, 64))
    assert z.shape == (2, 512) and torch.allclose(z.view(2, -1, 8).sum(-1), torch.ones(2, 64))


def test_goal_conditioned_latents_carry_the_goal():
    model = TDMPC2(
        state_encoder(6, latent_dim=32, hidden_dim=16),
        action_dim=2,
        latent_dim=32,
        mlp_dim=32,
        goal_conditioned=True,
        episodic=True,
    )
    obs, actions, rewards = batch()
    goal = torch.randn(8, 1, 6).expand(-1, 4, -1)
    out = model(torch.cat([obs, goal], dim=2), actions, rewards)
    assert all(v.isfinite() for v in out.values())
    z = model.encode(torch.cat([obs, goal], dim=2))
    assert z.shape == (8, 4, 64)
    assert torch.allclose(z[..., :32], model._encode(obs)) and torch.allclose(z[..., 32:], model._encode(goal))
    # dynamics move the observation's half and carry the goal's; the heads read both
    assert torch.equal(model.rollout(z[:, :1], actions)[..., 32:], z[:, :1, 32:].expand(-1, 4, -1))
    assert model.reward[0][0].in_features == model.qs[0][0][0].in_features == 66
    assert model.policy[0][0].in_features == model.termination[0][0].in_features == 64
    planner = MPPI(horizon=3, action_dim=2, num_samples=32, num_elites=4, num_pi_trajs=4, iterations=2)
    assert planner(model.eval(), z[:, :1]).shape == (8, 3, 2)


def test_episode_discount():
    assert TDMPC2.episode_discount(1000) == 0.995
    assert TDMPC2.episode_discount(10) == 0.95
    assert TDMPC2.episode_discount(100) == pytest.approx(0.95)
    assert TDMPC2.episode_discount(250) == pytest.approx(0.98)


class Bandit(TDMPC2):
    """Return is ``-||sum(actions) - target||^2``."""

    def __init__(self, target):
        super().__init__(state_encoder(2, latent_dim=8, hidden_dim=8), action_dim=2, latent_dim=8, mlp_dim=8)
        self.target = target

    def estimate_value(self, z, actions):
        return -(actions.sum(1) - self.target.repeat_interleave(len(z) // len(self.target), 0)).square().sum(-1)


def test_mppi_maximizes_value_and_warm_starts():
    target = torch.tensor([[0.5, -1.0], [-0.8, 0.3]])
    model = Bandit(target).eval()
    planner = MPPI(
        horizon=3, action_dim=2, num_samples=256, num_elites=32, num_pi_trajs=8, iterations=8, temperature=20
    )
    plan = planner(model, torch.zeros(2, 1, 8), generator=torch.Generator().manual_seed(0))
    assert plan.shape == (2, 3, 2) and plan.abs().le(1).all()
    assert torch.allclose(plan.sum(1), target, atol=0.1)
    assert planner.mean.shape == (2, 3, 2)
    planner.reset()
    assert planner.mean is None


def test_mppi_with_tdmpc2():
    model = tdmpc2(episodic=True).eval()
    planner = MPPI(horizon=3, action_dim=2, num_samples=32, num_elites=4, num_pi_trajs=4, iterations=2)
    plan = planner(model, model.encode(torch.randn(3, 1, 6)), explore=True)
    assert plan.shape == (3, 3, 2) and plan.abs().le(1).all()
