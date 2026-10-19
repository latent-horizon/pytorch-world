import torch
from torch import nn

from torch_world.losses import SIGReg
from torch_world.models import LeVJEPA, LeVJEPAEncoder


def make_encoder(**kwargs):
    return LeVJEPAEncoder(img_size=16, patch_size=4, embed_dim=24, depth=2, num_heads=2, num_frames=4, **kwargs)


def make_model(lamb=0.02, **kwargs):
    return LeVJEPA(make_encoder(**kwargs), nn.Linear(24, 8), lamb=lamb, sigreg=SIGReg(64))


def test_token_drop_in_training_only():
    encoder = make_encoder(token_drop=0.75)
    x = torch.randn(2, 3, 4, 16, 16)
    assert encoder(x).shape == (2, 1 + 16, 24)
    assert encoder.eval()(x).shape == (2, 1 + 64, 24)


def test_local_views_use_their_own_grid():
    encoder = make_encoder(token_drop=0.0)
    assert encoder(torch.randn(2, 3, 4, 8, 8)).shape == (2, 1 + 16, 24)


def test_block_causal_patches_never_see_later_frames():
    encoder = make_encoder().eval()
    x = torch.randn(2, 3, 4, 16, 16)
    y = x.clone()
    y[:, :, -1] = torch.randn_like(y[:, :, -1])
    a, b = encoder(x), encoder(y)
    assert torch.allclose(a[:, 1:49], b[:, 1:49], atol=1e-6)
    assert not torch.allclose(a[:, 0], b[:, 0])


def test_full_attention_lets_patches_see_later_frames():
    encoder = make_encoder(block_causal=False).eval()
    x = torch.randn(2, 3, 4, 16, 16)
    y = x.clone()
    y[:, :, -1] = torch.randn_like(y[:, :, -1])
    assert not torch.allclose(encoder(x)[:, 1:49], encoder(y)[:, 1:49])


def test_loss_combines_terms():
    out = make_model(lamb=0.3)(torch.randn(16, 3, 4, 16, 16), torch.randn(16, 2, 3, 4, 8, 8))
    assert torch.allclose(out["loss"], out["pred_loss"] + 0.3 * out["sigreg_loss"])


def test_views_identical_to_global_have_zero_prediction_loss():
    model = make_model().eval()
    x = torch.randn(16, 3, 4, 16, 16)
    assert model(x, torch.stack([x, x], dim=1))["pred_loss"] < 1e-12


def test_prediction_targets_are_the_global_projection():
    model = make_model().eval()
    g, local = torch.randn(8, 3, 4, 16, 16), torch.randn(8, 3, 3, 4, 8, 8)
    z = torch.stack([model.projector(model.encode(v)) for v in [g, *local.unbind(1)]])
    expected = (z[0] - z).square().mean()
    assert torch.allclose(model(g, local)["pred_loss"], expected, atol=1e-6)
    assert torch.allclose(model(g, list(local.unbind(1)))["pred_loss"], expected, atol=1e-6)


def test_gradients_reach_encoder():
    model = make_model()
    model(torch.randn(8, 3, 4, 16, 16), torch.randn(8, 2, 3, 4, 8, 8))["loss"].backward()
    assert model.encoder.cls_token.grad.abs().sum() > 0
    assert model.encoder.patch_embed.proj.weight.grad.abs().sum() > 0
