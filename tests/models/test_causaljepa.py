import itertools

import torch

from torch_world.models import CausalJEPA, SlotEncoder
from torch_world.models.jepa._causaljepa import slot_distances
from torch_world.planning import CEM


def encoder():
    return SlotEncoder(img_size=16, num_slots=3, slot_dim=16, channels=8, iters=2)


def causaljepa(**kwargs):
    kwargs = {"slot_dim": 16, "depth": 2, "num_heads": 2, "head_dim": 8, "mlp_dim": 32, "dropout": 0.0, **kwargs}
    return CausalJEPA(encoder(), action_dim=2, **kwargs)


def test_slot_encoder_shapes_and_reconstruction():
    enc = encoder()
    assert enc(torch.rand(4, 3, 16, 16)).shape == (4, 3, 16)
    assert enc(torch.rand(2, 5, 3, 16, 16)).shape == (2, 5, 3, 16)
    out = enc(torch.rand(2, 5, 3, 16, 16), reconstruct=True)
    assert out["recon"].shape == (10, 3, 16, 16) and torch.allclose(out["masks"].sum(1), torch.ones(10, 1, 16, 16))
    out["loss"].backward()
    assert enc.queries.grad.abs().sum() > 0 and enc.cnn[0].weight.grad.abs().sum() > 0


def test_slot_tracking():
    enc, video = encoder().eval(), torch.rand(2, 3, 3, 16, 16)
    with torch.no_grad():
        enc.track = None
        fresh = enc(video)
        enc.track = "propagate"
        prop = enc(video)
        enc.track = "match"
        matched = enc(video)
    assert torch.allclose(fresh[:, 0], prop[:, 0]) and not torch.allclose(fresh[:, 1:], prop[:, 1:])
    # matching only reorders the slots of each frame, toward the previous frame
    assert torch.allclose(matched.sort(2).values, fresh.sort(2).values) and torch.equal(matched[:, 0], fresh[:, 0])
    gap = lambda z: (z[:, 1] - z[:, 0]).square().mean((1, 2))
    assert (gap(matched) <= gap(fresh) + 1e-6).all()
    perm = torch.tensor([2, 0, 1])
    cost, reordered = slot_distances(fresh[:, 1, perm], fresh[:, 1])
    assert torch.allclose(cost, torch.zeros(2), atol=1e-6) and torch.allclose(reordered, fresh[:, 1])


def test_loss_components_and_gradients():
    model = causaljepa()
    out = model(torch.rand(4, 4, 3, 16, 16), torch.randn(4, 4, 2))
    assert torch.allclose(out["loss"], out["history_loss"] + out["future_loss"])
    out["loss"].backward()
    assert all(p.grad is None for p in model.encoder.parameters())  # frozen object encoder
    assert model.action_encoder.weight.grad.abs().sum() > 0 and model.mask_token.grad.abs().sum() > 0
    slots = torch.randn(4, 4, 3, 16)
    no_mask = causaljepa(max_masked=0)(slots, torch.randn(4, 4, 2))
    assert no_mask["history_loss"] == 0 and no_mask["future_loss"] > 0
    proprio = causaljepa(proprio_dim=4)(slots, torch.randn(4, 4, 2), torch.randn(4, 4, 4))
    assert torch.allclose(proprio["loss"], proprio["history_loss"] + proprio["future_loss"] + proprio["proprio_loss"])


def test_masked_object_is_hidden_after_the_anchor():
    model = causaljepa().eval()
    z, actions = torch.randn(2, 3, 3, 16), torch.randn(2, 3, 2)
    masked = torch.tensor([[True, False, False]] * 2)
    z2 = z.clone()
    z2[:, 1:, 0] = torch.randn(2, 2, 16)  # change the hidden object after its first step
    with torch.no_grad():
        a, b = model.predict(z, actions, masked=masked), model.predict(z2, actions, masked=masked)
        assert torch.allclose(a, b, atol=1e-5)  # nothing of the hidden history leaks in
        c = model.predict(z2, actions)
    assert a.shape == (2, 4, 3, 16) and not torch.allclose(a, c)
    z3 = z.clone()
    z3[:, 1:, 1] = torch.randn(2, 2, 16)  # the other objects are what the hidden one is inferred from
    with torch.no_grad():
        assert not torch.allclose(a[:, 1:, 0], model.predict(z3, actions, masked=masked)[:, 1:, 0])


def test_objects_are_permutation_equivariant():
    model = causaljepa().eval()
    z, actions, perm = torch.randn(2, 3, 3, 16), torch.randn(2, 3, 2), torch.tensor([2, 0, 1])
    with torch.no_grad():
        a, b = model.predict(z, actions), model.predict(z[:, :, perm], actions)
    assert torch.allclose(a[:, :, perm], b, atol=1e-5)


def test_rollout_and_matched_cost():
    model = causaljepa().eval()
    z = model.encode(torch.rand(2, 1, 3, 16, 16))
    with torch.no_grad():
        out = model.rollout(z, torch.randn(2, 5, 2))
    assert out.shape == (2, 6, 3, 16) and torch.equal(out[:, :1], z)
    goal = torch.randn(2, 3, 16)
    perm = torch.tensor([1, 2, 0])
    assert torch.allclose(model.cost(goal[:, perm][:, None], goal), torch.zeros(2), atol=1e-6)
    pairs = (out[:, -1, :, None] - goal[:, None]).square().mean(-1)
    brute = torch.stack([pairs[:, range(3), list(p)].mean(-1) for p in itertools.permutations(range(3))]).amin(0)
    assert torch.allclose(model.cost(out, goal), brute)
    assert torch.allclose(causaljepa(match=False).cost(out, goal), (out[:, -1] - goal).square().flatten(1).mean(1))


def test_cem_plans_with_rollout():
    model = causaljepa(proprio_dim=None).eval()
    z = model.encode(torch.rand(2, 1, 3, 16, 16))
    plan = CEM(horizon=3, action_dim=2, num_samples=16, num_elites=4, iterations=2, bounds=(-1, 1))(model, z, z[:, 0])
    assert plan.shape == (2, 3, 2) and plan.abs().max() <= 1
