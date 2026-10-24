import torch
from torch import nn

from torch_world.losses import SIGReg
from torch_world.models.jepa._hwm import HWM
from torch_world.models.jepa._lewm import LeWM
from torch_world.nn.transformer import AdaLNBlock

STRIDE, HISTORY = 2, 3


def hwm():
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
    model = HWM(low, 2, stride=STRIDE, macro_dim=3, embed_dim=32, history=HISTORY, depth=1, num_heads=2, head_dim=8)
    for m in model.modules():  # adaLN-zero ignores actions at init; give them an effect
        if isinstance(m, AdaLNBlock):
            nn.init.normal_(m.modulation[-1].weight, std=0.1)
    return model


def batch(b=4):
    T = HISTORY * STRIDE + 1
    return torch.randn(b, T, 3, 8, 8), torch.randn(b, T, 2)


def test_loss_is_finite_and_sums_terms():
    model = hwm()
    out = model(*batch())
    assert all(v.isfinite() for v in out.values())
    total = out["low_loss"] + out["rollout_loss"] + model.low.lamb * out["sigreg_loss"] + out["high_loss"]
    assert torch.allclose(out["loss"], total)


def test_gradients_reach_both_levels_and_high_level_is_stopped():
    model = hwm()
    pixels, actions = batch()
    out = model(pixels, actions)
    out["loss"].backward()
    assert model.low.encoder[1].weight.grad.abs().sum() > 0
    assert model.high.proj.weight.grad.abs().sum() > 0
    assert model.macro_encoder[0][0].weight.grad.abs().sum() > 0
    model.zero_grad()
    model(pixels, actions)["high_loss"].backward()
    assert model.low.encoder[1].weight.grad is None or model.low.encoder[1].weight.grad.abs().sum() == 0


def test_both_levels_predict():
    model = hwm().eval()
    pixels, actions = batch(2)
    z = model.encode(pixels)
    assert z.shape == (2, 7, 32)
    assert model.predict(z[:, :2], actions[:, :2]).shape == (2, 2, 32)
    macro = model.encode_macro(actions[:, :6])
    assert macro.shape == (2, 3, 3)
    assert model.high.predict(z[:, ::STRIDE][:, :3], macro).shape == (2, 3, 32)
    assert model.high.rollout(z[:, :1], torch.randn(2, 5, 3)).shape == (2, 6, 32)
    assert torch.equal(model.lift(z), z)
    assert model.cost(z, z[:, 0]).shape == (2,)


def test_macro_actions_are_standardized():
    model = hwm().train()
    macro = model.encode_macro(torch.randn(64, 6, 2)).flatten(0, 1)
    assert torch.allclose(macro.mean(0), torch.zeros(3), atol=1e-5)
    assert torch.allclose(macro.std(0, unbiased=False), torch.ones(3), atol=1e-2)
