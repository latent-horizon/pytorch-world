import torch

from torch_world.models.base import WorldModel
from torch_world.models.jepa._pldm import PLDM, PLDMEncoder
from torch_world.planning import CEM


def pldm(**kwargs):
    return PLDM(PLDMEncoder(16, latent_dim=32, channels=(8, 8)), action_dim=2, latent_dim=32, idm_dim=64, **kwargs)


def test_encoder_shape():
    assert PLDMEncoder(16, latent_dim=32, channels=(8, 8))(torch.randn(5, 3, 16, 16)).shape == (5, 32)
    assert PLDMEncoder(64, latent_dim=24, in_channels=2)(torch.randn(2, 2, 64, 64)).shape == (2, 24)


def test_loss_and_components():
    model = pldm()
    out = model(torch.randn(4, 6, 3, 16, 16), torch.randn(4, 6, 2))
    parts = ["pred_loss", "var_loss", "cov_loss", "smooth_loss", "idm_loss"]
    assert all(out[k].dim() == 0 and torch.isfinite(out[k]) for k in ["loss", *parts])
    weights = [1, model.var_coeff, model.cov_coeff, model.smooth_coeff, model.idm_coeff]
    assert torch.allclose(out["loss"], sum(w * out[k] for w, k in zip(weights, parts)))


def test_gradients_reach_encoder_predictor_and_idm():
    model = pldm()
    model(torch.randn(4, 5, 3, 16, 16), torch.randn(4, 5, 2))["loss"].backward()
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert model.predictor.weight_ih.grad.abs().sum() > 0
    assert model.idm[0].weight.grad.abs().sum() > 0


def test_each_term_trains_the_encoder():
    for term in ["pred_loss", "var_loss", "cov_loss", "smooth_loss", "idm_loss"]:
        model = pldm()
        model(torch.randn(4, 5, 3, 16, 16), torch.randn(4, 5, 2))[term].backward()
        assert model.encoder[0].weight.grad.abs().sum() > 0, term


def test_shapes_and_normalized_latents():
    model = pldm()
    z = model.encode(torch.randn(3, 4, 3, 16, 16))
    assert z.shape == (3, 4, 32)
    assert torch.allclose(z.mean(-1), torch.zeros(3, 4), atol=1e-5)
    assert model.predict(z, torch.randn(3, 4, 2)).shape == (3, 4, 32)


def test_rollout_is_causal_and_matches_the_base_rollout():
    model = pldm().eval()
    z, actions = torch.randn(2, 2, 32), torch.randn(2, 6, 2)
    out = model.rollout(z, actions)
    assert out.shape == (2, 7, 32)
    assert torch.equal(out[:, :2], z)
    assert torch.allclose(out, WorldModel.rollout(model, z, actions), atol=1e-6)
    changed = actions.clone()
    changed[:, 4:] = 0
    assert torch.allclose(model.rollout(z, changed)[:, :5], out[:, :5])
    assert not torch.allclose(model.rollout(z, changed)[:, 5:], out[:, 5:])


def test_training_unrolls_open_loop_from_the_first_frame():
    model = pldm()
    pixels, actions = torch.randn(4, 5, 3, 16, 16), torch.randn(4, 5, 2)
    z = model.encode(pixels)
    expected = (model.rollout(z[:, :1], actions[:, :4])[:, 1:] - z[:, 1:]).square().mean()
    assert torch.allclose(model(pixels, actions)["pred_loss"], expected)


def test_cem_plans_with_path_cost():
    model = pldm().eval()
    planner = CEM(horizon=4, action_dim=2, num_samples=16, num_elites=4, iterations=2, bounds=(-1, 1), objective="path")
    z = model.encode(torch.rand(2, 1, 3, 16, 16))
    goal = model.encode(torch.rand(2, 1, 3, 16, 16))[:, 0]
    plan = planner(model, z, goal, generator=torch.Generator().manual_seed(0))
    assert plan.shape == (2, 4, 2)
    assert plan.abs().max() <= 1
