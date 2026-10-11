import pytest
import torch
from torch import nn

from torch_world.losses import SIGReg
from torch_world.models import DINOWM, LeWM, RoboJEPA


class Patches(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, 4, 4)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


def lewm():
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 32))
    return LeWM(
        encoder,
        action_dim=2,
        embed_dim=32,
        history=3,
        depth=2,
        num_heads=2,
        head_dim=8,
        mlp_dim=64,
        proj_dim=64,
        sigreg=SIGReg(32),
    )


def dinowm():
    return DINOWM(
        Patches(),
        num_patches=4,
        embed_dim=16,
        action_dim=2,
        proprio_dim=2,
        history=3,
        depth=2,
        num_heads=2,
        head_dim=8,
        mlp_dim=32,
    )


def robojepa(**kwargs):
    return RoboJEPA(
        embed_dim=16,
        action_dim=3,
        state_dim=2,
        num_views=2,
        num_embodiments=2,
        predictor_dim=24,
        depth=2,
        num_heads=2,
        **kwargs,
    )


def test_lewm_loss_and_shapes():
    model = lewm()
    pixels, actions = torch.randn(4, 4, 3, 8, 8), torch.randn(4, 4, 2)
    out = model(pixels, actions)
    assert torch.allclose(out["loss"], out["pred_loss"] + 0.09 * out["sigreg_loss"])
    out["loss"].backward()
    assert model.encoder[1].weight.grad is not None
    assert model.encode(pixels).shape == (4, 4, 32)


def test_dinowm_encoder_frozen_and_action_dims_dropped():
    model = dinowm().train()
    assert not model.encoder.training
    z = model.encode(torch.randn(2, 4, 3, 8, 8), torch.randn(2, 4, 2))
    assert z.shape == (2, 4, 4, 26)
    assert model.predict(z[:, :3], torch.randn(2, 3, 2)).shape == (2, 3, 4, 26)
    out = model(torch.randn(2, 4, 3, 8, 8), torch.randn(2, 4, 2), torch.randn(2, 4, 2))
    out["loss"].backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert {"visual_loss", "proprio_loss"} <= out.keys()


@pytest.mark.parametrize("name", ["lewm", "dinowm"])
def test_prediction_is_causal(name):
    model = {"lewm": lewm, "dinowm": dinowm}[name]().eval()
    z = model.encode(torch.randn(2, 3, 3, 8, 8), *([torch.randn(2, 3, 2)] if name == "dinowm" else []))
    actions = torch.randn(2, 3, 2)
    pred = model.predict(z, actions)
    z2, a2 = z.clone(), actions.clone()
    z2[:, 2], a2[:, 2] = torch.randn_like(z2[:, 2]), torch.randn_like(a2[:, 2])
    pred2 = model.predict(z2, a2)
    assert torch.allclose(pred[:, :2], pred2[:, :2], atol=1e-5)
    assert not torch.allclose(pred[:, 2], pred2[:, 2])


@pytest.mark.parametrize("name", ["lewm", "dinowm"])
def test_rollout_uses_history_window(name):
    model = {"lewm": lewm, "dinowm": dinowm}[name]().eval()
    z = model.encode(torch.randn(2, 1, 3, 8, 8), *([torch.randn(2, 1, 2)] if name == "dinowm" else []))
    out = model.rollout(z, torch.randn(2, 6, 2))
    assert out.shape[1] == 7 and torch.equal(out[:, :1], z)
    assert model.cost(out, z[:, 0]).shape == (2,)


def test_robojepa_shapes_and_losses():
    model = robojepa(rollout_steps=3)
    z, actions, states = torch.randn(2, 5, 2, 3, 3, 16), torch.randn(2, 5, 3), torch.randn(2, 5, 2)
    out = model(z, actions, states, torch.tensor([0, 1]))
    assert torch.allclose(out["loss"], out["tf_loss"] + out["rollout_loss"])
    out["loss"].backward()
    assert model.action_proj.weight.grad[1].abs().sum() > 0
    assert model.predict(z, actions, states).shape == z.shape


def test_robojepa_causal_window():
    model = robojepa(window=2).eval()
    z, actions, states = torch.randn(1, 4, 2, 3, 3, 16), torch.randn(1, 4, 3), torch.randn(1, 4, 2)
    pred = model.predict(z, actions, states)
    z2 = z.clone()
    z2[:, 0] = torch.randn_like(z2[:, 0])
    pred2 = model.predict(z2, actions, states)
    assert not torch.allclose(pred[:, 1], pred2[:, 1])
    assert torch.allclose(pred[:, 2:], pred2[:, 2:], atol=1e-5)


def test_robojepa_embodiments_use_separate_projections():
    model = robojepa().eval()
    z, actions, states = torch.randn(1, 2, 2, 3, 3, 16), torch.randn(1, 2, 3), torch.randn(1, 2, 2)
    a = model.predict(z, actions, states, torch.tensor([0]))
    b = model.predict(z, actions, states, torch.tensor([1]))
    assert not torch.allclose(a, b)


def test_robojepa_rollout_pads_states():
    model = robojepa().eval()
    z = torch.randn(2, 1, 2, 3, 3, 16)
    out = model.rollout(z, torch.randn(2, 4, 3), torch.randn(2, 1, 2))
    assert out.shape == (2, 5, 2, 3, 3, 16)
