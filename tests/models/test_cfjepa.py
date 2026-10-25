import pytest
import torch
from torch import nn

from torch_world.losses import SIGReg
from torch_world.models.jepa._cfjepa import CFJEPA, GradientReversal
from torch_world.planning import CEM


def cfjepa(**kwargs):
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 32))
    kwargs = {"ctrl_dim": 20, **kwargs}
    model = CFJEPA(
        encoder, action_dim=2, embed_dim=32, history=3, depth=2, num_heads=2, head_dim=8, mlp_dim=64, proj_dim=64,
        inv_dim=32, sigreg=SIGReg(32), **kwargs,
    )  # fmt: skip
    for block in model.ctrl_predictor.blocks:  # adaLN-zero gates start closed: open them so the actions matter
        nn.init.normal_(block.modulation[-1].weight, std=0.1)
    return model


def test_loss_components_and_gradients():
    model = cfjepa(alpha=2.0, beta=0.5, lamb=0.1)
    out = model(torch.randn(4, 4, 3, 8, 8), torch.randn(4, 4, 2))
    assert all(torch.isfinite(v) for v in out.values())
    expected = out["pred_loss"] + 2.0 * out["inverse_loss"] + 0.5 * out["adversary_loss"] + 0.1 * out["sigreg_loss"]
    assert torch.allclose(out["loss"], expected)
    out["loss"].backward()
    for module in (model.encoder[1], model.action_encoder[0], model.inverse[0], model.adversary[0]):
        assert module.weight.grad.abs().sum() > 0
    assert model.unctrl_predictor.blocks[0].attn.qkv.weight.grad.abs().sum() > 0


def test_gradient_reversal():
    x = torch.randn(3, 4, requires_grad=True)
    GradientReversal.apply(x, 0.5).sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, -0.5))


def test_adversary_gradient_reaches_encoder_reversed():
    # the adversary descends its loss while the encoder ascends it
    model = cfjepa()
    pixels, actions = torch.randn(4, 4, 3, 8, 8), torch.randn(4, 4, 2)
    z = model.encode(pixels).detach().requires_grad_(True)
    zu = z[:, :4, 20:]
    pair = torch.cat([zu[:, :-1], zu[:, 1:]], -1)
    plain = torch.autograd.grad((model.adversary(pair) - actions[:, :3]).square().mean(), z)[0]
    reversed_ = torch.autograd.grad(
        (model.adversary(GradientReversal.apply(pair, 1.0)) - actions[:, :3]).square().mean(), z
    )[0]
    assert plain.abs().sum() > 0 and torch.allclose(reversed_, -plain)


def test_factored_dynamics():
    model = cfjepa().eval()
    z, a = torch.randn(2, 3, 32), torch.randn(2, 3, 2)
    with torch.no_grad():
        pred = model.predict(z, a)
        a2 = torch.randn_like(a)
        z2 = z.clone()
        z2[..., 20:] = torch.randn_like(z2[..., 20:])
        by_action, by_unctrl = model.predict(z, a2), model.predict(z2, a)
    assert pred.shape == (2, 3, 32)
    # actions move only z^c; z^u never reaches the controllable predictor
    assert torch.allclose(pred[..., 20:], by_action[..., 20:])
    assert not torch.allclose(pred[..., :20], by_action[..., :20])
    assert torch.allclose(pred[..., :20], by_unctrl[..., :20])
    assert not torch.allclose(pred[..., 20:], by_unctrl[..., 20:])


def test_causal_prediction():
    model = cfjepa().eval()
    z, a = torch.randn(2, 3, 32), torch.randn(2, 3, 2)
    z2, a2 = z.clone(), a.clone()
    z2[:, 2], a2[:, 2] = torch.randn(2, 32), torch.randn(2, 2)
    with torch.no_grad():
        assert torch.allclose(model.predict(z, a)[:, :2], model.predict(z2, a2)[:, :2], atol=1e-5)


def test_cost_ignores_uncontrollable_part():
    model = cfjepa()
    z, goal = torch.randn(4, 5, 32), torch.randn(4, 32)
    other = goal.clone()
    other[:, 20:] = torch.randn(4, 12)
    assert torch.allclose(model.cost(z, goal), model.cost(z, other))
    assert torch.allclose(model.cost(z, goal), (z[:, -1, :20] - goal[:, :20]).square().mean(1))
    with pytest.raises(ValueError):
        cfjepa(ctrl_dim=32)


def test_cem_plans_with_rollout():
    model = cfjepa().eval()
    z = model.encode(torch.randn(2, 1, 3, 8, 8))
    assert model.rollout(z, torch.randn(2, 4, 2)).shape == (2, 5, 32)
    plan = CEM(horizon=3, action_dim=2, num_samples=16, num_elites=4, iterations=2, bounds=(-1, 1))(model, z, z[:, 0])
    assert plan.shape == (2, 3, 2) and plan.abs().max() <= 1
