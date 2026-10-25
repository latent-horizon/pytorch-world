import pytest
import torch

from torch_world.models import DreamerV3
from torch_world.models.dreamer import two_hot, two_hot_inv
from torch_world.planning import CEM


def dreamer(**kwargs):
    small = {"img_size": 16, "cnn_depth": 4, "deter_dim": 16, "stoch": 4, "classes": 4, "hidden_dim": 16}
    return DreamerV3(action_dim=2, **{"mlp_layers": 1, **small, **kwargs})


def batch(B=3, T=4):
    return torch.rand(B, T + 1, 3, 16, 16), torch.rand(B, T, 2) * 2 - 1, torch.randn(B, T)


def test_world_model_losses_without_rewards():
    model = dreamer()
    obs, actions, _ = batch()
    out = model(obs, actions)
    assert set(out) == {"loss", "image_loss", "dyn_loss", "rep_loss", "kl"}
    assert torch.allclose(out["loss"], out["image_loss"] + 0.5 * out["dyn_loss"] + 0.1 * out["rep_loss"])
    assert out["dyn_loss"] >= 1 and out["rep_loss"] >= 1  # free bits
    out["loss"].backward()
    assert model.encoder[1].weight.grad.abs().sum() > 0 and model.gru.linear.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in [*model.actor.parameters(), *model.critic.parameters()])


def test_full_loss_and_disjoint_gradients():
    model = dreamer()
    out = model(*batch())
    assert all(v.isfinite() for v in out.values())
    world = out["image_loss"] + 0.5 * out["dyn_loss"] + 0.1 * out["rep_loss"] + out["reward_loss"]
    assert torch.allclose(out["loss"], world + out["continue_loss"] + out["critic_loss"] + out["actor_loss"])

    model.zero_grad()
    model(*batch())["actor_loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.actor.parameters())
    world_params = [p for g in model.param_groups()[:1] for p in g["params"]]
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in [*world_params, *model.critic.parameters()])

    model.zero_grad()
    model(*batch())["critic_loss"].backward()
    assert all(p.grad is None for p in [*world_params, *model.actor.parameters()])


def test_dynamics_gradient_reaches_actor_through_rewards():
    model = dreamer()
    for p in model.reward[-1].parameters():  # zero-initialized: give the reward head an opinion
        torch.nn.init.normal_(p, std=1.0)
    model(*batch())["actor_loss"].backward()
    assert model.actor[0].weight.grad.abs().sum() > 0
    reinforce = dreamer(actor_grad="reinforce")
    reinforce(*batch())["actor_loss"].backward()
    assert reinforce.actor[0].weight.grad is not None
    with pytest.raises(ValueError):
        dreamer(actor_grad="backprop")


def test_categoricals_unimix_and_straight_through():
    model = dreamer()
    logp = model._dist(torch.randn(2, 16) * 100)
    assert torch.allclose(logp.exp().sum(-1), torch.ones(2, 4))
    assert logp.exp().min() >= 0.01 / 4 - 1e-6  # 1% uniform
    logits = torch.randn(2, 16, requires_grad=True)
    s = model._draw(model._dist(logits), sample=True)
    assert torch.allclose(s.detach().unflatten(-1, (4, 4)).sum(-1), torch.ones(2, 4))
    assert torch.allclose(s.detach().max(-1).values, torch.ones(2)) and (s.detach() > 0.5).sum() == 8
    (s * torch.randn_like(s)).sum().backward()
    assert logits.grad.abs().sum() > 0


def test_shapes_rollout_and_filtering():
    model = dreamer().eval()
    obs, actions, _ = batch()
    with torch.no_grad():
        z = model.encode(obs, actions)
        assert z.shape == (3, 5, model.feat_dim) == (3, 5, 16 + 16)
        assert torch.allclose(model.step(z[:, 0], actions[:, 0], obs[:, 1]), z[:, 1])
        out = model.rollout(z[:, :1], actions)
        assert out.shape == (3, 5, 32) and torch.equal(out[:, :1], z[:, :1])
        assert model.decode(out).shape == (3, 5, 3, 16, 16)
        assert model.predict(z, torch.zeros(3, 5, 2)).shape == (3, 5, 32)
        assert model.act(z).shape == (3, 5, 2) and model.act(z, sample=True).abs().max() <= 1
        assert torch.equal(model.encode(obs, actions), z)  # the mode is deterministic


def test_prediction_is_causal_in_actions():
    model = dreamer().eval()
    obs, actions, _ = batch()
    other = actions.clone()
    other[:, 2:] = torch.randn_like(other[:, 2:])
    with torch.no_grad():
        z, z2 = model.rollout(model.encode(obs[:, :1]), actions), model.rollout(model.encode(obs[:, :1]), other)
    assert torch.equal(z[:, :3], z2[:, :3]) and not torch.allclose(z[:, 3:], z2[:, 3:])


def test_two_hot_round_trip_and_lambda_returns():
    bins = torch.linspace(-20, 20, 255)
    x = torch.tensor([-50.0, -1.3, 0.0, 0.7, 12.0, 1e9])
    target = two_hot(x, bins)
    assert torch.allclose(target.sum(-1), torch.ones(6)) and (target > 0).sum(-1).le(2).all()
    assert torch.allclose(two_hot_inv(target.clamp_min(1e-12).log(), bins)[:5], x[:5], rtol=1e-4, atol=1e-4)
    model = dreamer(discount=0.9, lam=1.0)
    rewards, conts, values = torch.ones(1, 3), torch.ones(1, 3), torch.tensor([[0.0, 0.0, 0.0, 10.0]])
    assert torch.allclose(model.lambda_returns(rewards, conts, values)[0, 0], torch.tensor(1 + 0.9 + 0.81 + 7.29))
    model.lam = 0.0  # one-step TD targets
    assert torch.allclose(model.lambda_returns(rewards, conts, values)[0], torch.tensor([1.0, 1.0, 10.0]))


def test_return_normalization_and_slow_critic():
    model = dreamer().train()
    model(*batch())
    assert model.ret_high >= model.ret_low
    low = model.ret_low.clone()
    model.eval()(*batch())
    assert torch.equal(model.ret_low, low)
    for p in model.critic.parameters():
        p.data.add_(1.0)
    before = [p.clone() for p in model.slow_critic.parameters()]
    model.update_target(0.98)
    for b, s, c in zip(before, model.slow_critic.parameters(), model.critic.parameters()):
        assert torch.allclose(s, 0.98 * b + 0.02 * c)


def test_param_groups_cover_every_trainable_parameter():
    model = dreamer()
    groups = model.param_groups()
    assert sum(len(g["params"]) for g in groups) == len([p for p in model.parameters() if p.requires_grad])
    assert groups[0]["lr"] == 1e-4 and groups[1]["lr"] == 3e-5 and groups[1]["eps"] == 1e-5
    model(*batch())["loss"].backward()
    assert len(model.clip_grad_norm()) == 3


def test_cem_plans_on_model_states():
    model = dreamer().eval()
    obs, _, _ = batch()
    z = model.encode(obs[:, :1])
    plan = CEM(horizon=3, action_dim=2, num_samples=16, num_elites=4, iterations=2, bounds=(-1, 1))(model, z, z[:, 0])
    assert plan.shape == (3, 3, 2) and plan.abs().max() <= 1


def test_cost_on_selects_part_of_the_state():
    z, goal = torch.zeros(2, 3, 32), torch.zeros(2, 32)
    goal[:, :16] = 1.0  # differs only in the deterministic part
    assert torch.equal(dreamer(cost_on="stoch").cost(z, goal), torch.zeros(2))
    assert torch.allclose(dreamer(cost_on="deter").cost(z, goal), torch.ones(2))
    assert torch.allclose(dreamer().cost(z, goal), torch.full((2,), 0.5))
    with pytest.raises(ValueError):
        dreamer(cost_on="pixels")
