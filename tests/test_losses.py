import torch

from torch_world.losses import SIGReg, info_nce


def reference_sigreg(proj, A, knots=17):
    t = torch.linspace(0, 3, knots)
    dt = 3 / (knots - 1)
    weights = torch.full((knots,), 2 * dt)
    weights[[0, -1]] = dt
    window = torch.exp(-t.square() / 2.0)
    x_t = (proj @ A).unsqueeze(-1) * t
    err = (x_t.cos().mean(-3) - window).square() + x_t.sin().mean(-3).square()
    return ((err @ (weights * window)) * proj.size(-2)).mean()


def test_sigreg_matches_reference():
    x = torch.randn(3, 64, 10)
    g = torch.Generator().manual_seed(7)
    A = torch.randn(10, 128, generator=torch.Generator().manual_seed(7))
    assert torch.allclose(SIGReg(128)(x, generator=g), reference_sigreg(x, A / A.norm(dim=0)), atol=1e-5)


def test_sigreg_prefers_isotropic_gaussian():
    sigreg = SIGReg(256)
    gaussian = sigreg(torch.randn(2048, 16))
    collapsed = sigreg(torch.randn(2048, 1).expand(-1, 16) * 0.1)
    shifted = sigreg(torch.randn(2048, 16) + 2)
    assert gaussian < 0.1 * collapsed and gaussian < 0.1 * shifted


def test_info_nce():
    a = torch.eye(8)
    assert info_nce(a, a, 0.01) < 1e-3
    b = torch.randn(8, 4)
    assert torch.allclose(info_nce(a[:, :4], b), info_nce(b, a[:, :4]))
