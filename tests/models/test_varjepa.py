import torch
from torch import nn

from torch_world.losses import SIGReg
from torch_world.models.jepa._varjepa import VariationalJEPA
from torch_world.planning import CEM


def varjepa():
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 32))
    model = VariationalJEPA(
        encoder, action_dim=2, embed_dim=32, history=3, depth=2, num_heads=2, head_dim=8, mlp_dim=64, proj_dim=64,
        sigreg=SIGReg(32),
    )  # fmt: skip
    for block in model.predictor.blocks:  # adaLN-zero gates start closed: open them so the context matters
        nn.init.normal_(block.modulation[-1].weight, std=0.1)
    return model


def test_loss_components_and_gradients():
    model = varjepa()
    out = model(torch.randn(4, 4, 3, 8, 8), torch.randn(4, 4, 2))
    assert all(torch.isfinite(v) for v in out.values())
    assert torch.allclose(out["loss"], out["nll_loss"] + 0.01 * out["kl_loss"] + 0.09 * out["sigreg_loss"])
    out["loss"].backward()
    assert model.encoder[0][1].weight.grad.abs().sum() > 0
    assert model.target_logvar.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.target_encoder.parameters())


def test_target_follows_ema():
    model = varjepa()
    with torch.no_grad():
        model.encoder[0][1].weight.add_(1.0)
    before = model.target_encoder[0][1].weight.clone()
    model.update_target(0.9)
    after = model.target_encoder[0][1].weight
    assert torch.allclose(after, 0.9 * before + 0.1 * model.encoder[0][1].weight)


def test_mean_samples_and_variance():
    model = varjepa().eval()
    z = model.encode(torch.randn(2, 3, 3, 8, 8))
    actions = torch.randn(2, 3, 2)
    with torch.no_grad():
        mean, logvar = model.distribution(z, actions)
        assert torch.equal(model.predict(z, actions), mean) and (logvar.exp() > 0).all()
        a = model.predict(z, actions, sample=True, generator=torch.Generator().manual_seed(0))
        b = model.predict(z, actions, sample=True, generator=torch.Generator().manual_seed(1))
        samples = model.sample(z[:, :1], torch.randn(2, 5, 2), num_samples=4)
    assert not torch.allclose(a, b)
    assert samples.shape == (2, 4, 6, 32) and samples[:, :, 1:].std(1).mean() > 0


def test_prediction_is_causal():
    model = varjepa().eval()
    z, actions = torch.randn(2, 3, 32), torch.randn(2, 3, 2)
    z2 = z.clone()
    z2[:, 2] = torch.randn_like(z2[:, 2])
    with torch.no_grad():
        pred, pred2 = model.predict(z, actions), model.predict(z2, actions)
    assert torch.allclose(pred[:, :2], pred2[:, :2], atol=1e-5) and not torch.allclose(pred[:, 2], pred2[:, 2])


def test_cem_plans_on_mean_and_samples():
    model = varjepa().eval()
    z = model.encode(torch.randn(2, 1, 3, 8, 8))
    cem = CEM(horizon=3, action_dim=2, num_samples=16, num_elites=4, iterations=2, bounds=(-1, 1))
    assert cem(model, z, z[:, 0]).shape == (2, 3, 2)
    assert cem(model, z, z[:, 0], sample=True).shape == (2, 3, 2)
