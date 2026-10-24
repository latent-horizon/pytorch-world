import torch
from torch import nn

from torch_world.losses import SIGReg
from torch_world.models.jepa._flowjepa import FlowJEPA
from torch_world.planning import CEM


def flowjepa(**kwargs):
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 32))
    model = FlowJEPA(
        encoder, action_dim=2, embed_dim=32, num_preds=3, depth=2, num_heads=2, head_dim=8, mlp_dim=64, proj_dim=64,
        steps=4, sigreg=SIGReg(32), **kwargs,
    )  # fmt: skip
    for block in model.predictor.blocks:  # adaLN-zero gates start closed: open them so the context matters
        nn.init.normal_(block.modulation[-1].weight, std=0.1)
    return model


def test_loss_components_and_gradients():
    model = flowjepa()
    out = model(torch.randn(4, 4, 3, 8, 8), torch.randn(4, 4, 2))
    assert all(torch.isfinite(v) for v in out.values())
    assert torch.allclose(out["loss"], out["flow_loss"] + 0.1 * out["sigreg_loss"])
    out["loss"].backward()
    assert model.encoder[1].weight.grad.abs().sum() > 0
    assert model.action_encoder[0].weight.grad.abs().sum() > 0


def test_future_tokens_see_only_past_actions():
    model = flowjepa().eval()
    z, a, x = torch.randn(2, 1, 32), model.action_encoder(torch.randn(2, 3, 2)), torch.randn(2, 3, 32)
    tau = torch.rand(2)
    a2 = a.clone()
    a2[:, 2] = torch.randn_like(a2[:, 2])
    with torch.no_grad():
        v, v2 = model.velocity(x, tau, z, a), model.velocity(x, tau, z, a2)
    # bidirectional future tokens: a_{t+2} reaches x_3 directly and the others through it
    assert not torch.allclose(v[:, 2], v2[:, 2])
    causal = flowjepa(causal=True).eval()
    with torch.no_grad():
        v, v2 = causal.velocity(x, tau, z, a), causal.velocity(x, tau, z, a2)
    assert torch.allclose(v[:, :2], v2[:, :2], atol=1e-5) and not torch.allclose(v[:, 2], v2[:, 2])


def test_shapes_and_samples_differ():
    model = flowjepa().eval()
    z = model.encode(torch.randn(2, 1, 3, 8, 8))
    actions = torch.randn(2, 5, 2)
    with torch.no_grad():
        out = model.rollout(z, actions)
        assert out.shape == (2, 6, 32) and torch.equal(out[:, :1], z)
        a = model.rollout(z, actions, generator=torch.Generator().manual_seed(0))
        b = model.rollout(z, actions, generator=torch.Generator().manual_seed(0))
        c = model.rollout(z, actions, generator=torch.Generator().manual_seed(1))
        samples = model.sample(z, actions, num_samples=4)
    assert torch.equal(a, b) and not torch.allclose(a, c)
    assert samples.shape == (2, 4, 6, 32) and samples[:, :, 1:].std(1).mean() > 0
    assert model.predict(z, actions[:, :3]).shape == (2, 3, 32)


def test_current_source_centers_noise():
    model = flowjepa(source="current", source_std=0.5)
    z, noise = torch.randn(2, 1, 32), torch.randn(2, 3, 32)
    assert torch.allclose(model.source_latents(z, noise), z + 0.5 * noise)


def test_cem_plans_with_rollout():
    model = flowjepa().eval()
    z = model.encode(torch.randn(2, 1, 3, 8, 8))
    plan = CEM(horizon=3, action_dim=2, num_samples=16, num_elites=4, iterations=2, bounds=(-1, 1))(model, z, z[:, 0])
    assert plan.shape == (2, 3, 2) and plan.abs().max() <= 1
