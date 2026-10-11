import torch
from torch import nn

from torch_world.losses import SIGReg
from torch_world.models import LeJEPA


def make_model(lamb=0.05):
    return LeJEPA(nn.Sequential(nn.Flatten(), nn.Linear(12, 16)), nn.Linear(16, 8), lamb=lamb, sigreg=SIGReg(64))


def test_loss_combines_terms():
    model = make_model(0.3)
    out = model(torch.randn(32, 4, 3, 2, 2))
    assert torch.allclose(out["loss"], 0.7 * out["pred_loss"] + 0.3 * out["sigreg_loss"])


def test_identical_views_have_zero_prediction_loss():
    x = torch.randn(32, 3, 2, 2)
    out = make_model()(torch.stack([x, x, x], dim=1))
    assert out["pred_loss"] < 1e-12


def test_prediction_targets_are_global_view_centers():
    model = make_model()
    views = [torch.randn(8, 3, 2, 2) for _ in range(3)]
    z = torch.stack([model.projector(model.encoder(v)) for v in views])
    expected = (z[:2].mean(0) - z).square().mean()
    assert torch.allclose(model(views, num_global=2)["pred_loss"], expected)


def test_views_with_different_resolutions():
    model = LeJEPA(nn.Sequential(nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(12, 16)), nn.Linear(16, 8))
    out = model([torch.randn(8, 3, 8, 8), torch.randn(8, 3, 8, 8), torch.randn(8, 3, 4, 4)], num_global=2)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
