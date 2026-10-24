import torch
from torch import nn

from torch_world.losses import SIGReg
from torch_world.models.jepa._hjepa import HJEPA
from torch_world.models.jepa._lewm import LeWM
from torch_world.nn.transformer import AdaLNBlock

STRIDE, HISTORY = 2, 3


def hjepa():
    low = LeWM(
        nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 32)),
        action_dim=2,
        embed_dim=32,
        history=2,
        depth=1,
        num_heads=2,
        head_dim=8,
        mlp_dim=64,
        proj_dim=64,
        sigreg=SIGReg(16),
    )
    model = HJEPA(
        low,
        stride=STRIDE,
        macro_dim=4,
        embed_dim=32,
        encoder_dim=48,
        history=HISTORY,
        depth=1,
        num_heads=2,
        head_dim=8,
        sigreg=SIGReg(16),
    )
    for m in model.modules():  # adaLN-zero ignores actions at init; give them an effect
        if isinstance(m, AdaLNBlock):
            nn.init.normal_(m.modulation[-1].weight, std=0.1)
    return model


def batch(b=4):
    T = HISTORY * STRIDE + 1
    return torch.randn(b, T, 3, 8, 8), torch.randn(b, T, 2)


def test_loss_is_finite_and_sums_levels():
    model = hjepa()
    out = model(*batch())
    assert all(v.isfinite() for v in out.values())
    total = out["low_loss"] + out["high_loss"] + model.low.lamb * (out["low_sigreg_loss"] + out["high_sigreg_loss"])
    assert torch.allclose(out["loss"], total)


def test_level_two_loss_trains_level_one_end_to_end():
    model = hjepa()
    model(*batch())["high_loss"].backward()
    assert model.low.encoder[1].weight.grad.abs().sum() > 0
    assert model.low.action_encoder[0].weight.grad.abs().sum() > 0
    assert model.state_encoder[0].weight.grad.abs().sum() > 0
    assert model.high.proj.weight.grad.abs().sum() > 0


def test_both_levels_predict():
    model = hjepa().eval()
    pixels, actions = batch(2)
    z = model.encode(pixels)
    assert z.shape == (2, 7, 32)
    assert model.predict(z[:, :2], actions[:, :2]).shape == (2, 2, 32)
    z2 = model.lift(z[:, ::STRIDE])
    assert z2.shape == (2, 4, 32)
    macro = model.encode_macro(actions[:, :6])
    assert macro.shape == (2, 3, 4)
    assert model.high.predict(z2[:, :3], macro).shape == (2, 3, 32)
    assert model.high.rollout(z2[:, :1], torch.randn(2, 4, 4)).shape == (2, 5, 32)
