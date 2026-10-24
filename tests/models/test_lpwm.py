import pytest
import torch
from torch import nn

from torch_world.losses import RDMReg
from torch_world.models import LpWM
from torch_world.models.jepa._lpwm import rep_relu
from torch_world.planning import CEM


def lpwm(history=3):
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 32))
    return LpWM(
        encoder,
        action_dim=2,
        embed_dim=32,
        history=history,
        depth=2,
        num_heads=2,
        head_dim=8,
        mlp_dim=64,
        proj_dim=64,
        rdmreg=RDMReg(64),
    )


def test_shapes_loss_and_gradients():
    torch.manual_seed(0)
    model = lpwm()
    pixels, actions = torch.randn(16, 4, 3, 8, 8), torch.randn(16, 4, 2)
    out = model(pixels, actions)
    assert all(torch.isfinite(v) for v in out.values())
    assert torch.allclose(out["loss"], out["pred_loss"] + 10.0 * out["rdmreg_loss"])
    out["loss"].backward()
    assert model.encoder[1].weight.grad.abs().sum() > 0
    assert all(p.grad is not None for p in model.parameters())  # the action path is zero at init (adaLN-zero)
    z = model.encode(pixels)
    assert z.shape == (16, 4, 32)
    assert model.predict(z[:, :3], actions[:, :3]).shape == (16, 3, 32)


def test_latents_are_nonnegative_and_sparse():
    torch.manual_seed(0)
    model = lpwm().eval()
    z = model.encode(torch.randn(8, 4, 3, 8, 8))
    pred = model.predict(z[:, :3], torch.randn(8, 3, 2))
    for code in (z, pred):
        assert (code >= 0).all()
        assert 0 < (code == 0).float().mean() < 1


def test_rep_relu_is_relu_forward_gelu_backward():
    x = torch.linspace(-3, 3, 13, requires_grad=True)
    y = rep_relu(x)
    assert torch.equal(y, x.relu())
    y.sum().backward()
    gelu_grad = torch.autograd.grad(nn.functional.gelu(x).sum(), x)[0]
    assert torch.allclose(x.grad, gelu_grad)
    assert x.grad[0] != 0


def test_rdmreg_target_and_distance():
    torch.manual_seed(0)
    reg = RDMReg(256)
    target = reg.sample((20000, 16))
    assert (target >= 0).all()
    assert abs((target == 0).float().mean().item() - 0.5) < 0.02  # rectified zero-mean Laplace: half the units off
    assert abs(target.var().item() - 0.375) < 0.02  # E[x^2] = 1/2 and E[x] = sigma / 2 = 1 / (2 sqrt 2)
    matched, dense = reg(reg.sample((512, 4, 16))), reg(torch.randn(512, 4, 16))
    assert matched < 0.01 < dense
    gaussian = RDMReg(256, p=2.0, rectify=False).sample((20000, 4))
    assert abs(gaussian.std().item() - 1) < 0.03 and (gaussian < 0).any()


def test_training_drives_codes_toward_sparse_target():
    torch.manual_seed(0)
    model = lpwm()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pixels, actions = torch.randn(64, 4, 3, 8, 8), torch.randn(64, 4, 2)
    first = model(pixels, actions)["rdmreg_loss"].item()
    for _ in range(100):
        out = model(pixels, actions)
        opt.zero_grad()
        (out["rdmreg_loss"]).backward()
        opt.step()
    assert out["rdmreg_loss"].item() < first / 4
    active = (model.eval().encode(pixels) > 0).float().mean().item()
    assert 0.3 < active < 0.7


@pytest.mark.parametrize("history", [1, 3])
def test_rollout_and_cem_planning(history):
    torch.manual_seed(0)
    model = lpwm(history).eval()
    z = model.encode(torch.randn(2, 1, 3, 8, 8))
    out = model.rollout(z, torch.randn(2, 5, 2))
    assert out.shape == (2, 6, 32) and torch.equal(out[:, :1], z) and (out >= 0).all()
    goal = model.encode(torch.randn(2, 1, 3, 8, 8))[:, 0]
    plan = CEM(horizon=4, action_dim=2, num_samples=32, num_elites=4, iterations=3, bounds=(-1, 1))(model, z, goal)
    assert plan.shape == (2, 4, 2) and plan.abs().max() <= 1
