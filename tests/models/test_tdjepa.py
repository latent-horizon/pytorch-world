import math

import torch

from torch_world import evaluation
from torch_world.envs import TwoRooms
from torch_world.models.jepa import TDJEPA, drq_encoder


def tdjepa(**kwargs):
    torch.manual_seed(0)
    return TDJEPA(
        drq_encoder(16, img_size=16), drq_encoder(8, hidden_layers=1, hidden_dim=16, img_size=16), action_dim=2,
        phi_dim=16, task_dim=8, hidden_dim=32, actor_dim=16, **kwargs,
    )  # fmt: skip


def grads(module):
    return sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)


def test_loss_components_and_gradients():
    model = tdjepa()
    out = model(torch.rand(6, 2, 3, 16, 16), torch.rand(6, 2, 2) * 2 - 1)
    assert all(torch.isfinite(v) for v in out.values())
    parts = out["td_phi_loss"] + out["td_psi_loss"] + out["ortho_loss"] + out["actor_loss"]
    assert torch.allclose(out["loss"], parts)
    out["loss"].backward()
    for name in ("phi_encoder", "psi_encoder", "phi_predictor", "psi_predictor", "actor"):
        assert grads(getattr(model, name)) > 0, name
    assert all(p.grad is None for p in model.targets.parameters())


def test_each_loss_reaches_only_its_parameters():
    model = tdjepa(lamb=0.0)
    model(torch.rand(6, 2, 3, 16, 16), torch.rand(6, 2, 2))["td_phi_loss"].backward()
    assert grads(model.phi_encoder) > 0 and grads(model.phi_predictor) > 0
    assert grads(model.psi_encoder) == grads(model.psi_predictor) == grads(model.actor) == 0  # psi- is a target
    model.zero_grad(set_to_none=True)
    model(torch.rand(6, 2, 3, 16, 16), torch.rand(6, 2, 2))["actor_loss"].backward()
    assert grads(model.actor) > 0
    assert sum(grads(getattr(model, n)) for n in ("phi_encoder", "psi_encoder", "phi_predictor", "psi_predictor")) == 0


def test_orthonormality_matches_algorithm_1():
    x = torch.randn(5, 3)
    B = len(x)
    off = sum((x[i] @ x[j]) ** 2 for i in range(B) for j in range(B) if i != j) / (2 * B * (B - 1))
    assert torch.allclose(TDJEPA.orthonormality(x), off - (x * x).sum(1).mean(), atol=1e-5)
    # an orthonormal batch scores lower than a collapsed one
    assert TDJEPA.orthonormality(torch.eye(4)) < TDJEPA.orthonormality(torch.ones(4, 4) / 2)


def test_td_target_is_bootstrapped_successor_features():
    model = tdjepa(p_goal=0.0)
    s, a = torch.rand(4, 2, 3, 16, 16), torch.rand(4, 2, 2)
    model.eval()
    with torch.no_grad():
        next_psi = model.psi(s[:, 1], target=True)
        bootstrap = model.targets["phi_predictor"](model.phi(s[:, 1], target=True), torch.zeros(4, 2), torch.ones(4, 8))
    # gamma = 0: the target is psi-(s'), so predicting exactly it gives zero loss
    model.discount = 0.0
    model.phi_predictor.forward = lambda x, act, z: next_psi.expand(2, -1, -1)
    assert model(s, a)["td_phi_loss"].abs() < 1e-6
    # gamma > 0: the target adds the target predictor at (phi-(s'), a' ~ pi, z), which is nonzero
    model.discount = 0.9
    assert model(s, a)["td_phi_loss"] > 0 and bootstrap.abs().sum() > 0


def test_tasks_and_embeddings_on_the_sphere():
    model = tdjepa().eval()
    z = model.sample_tasks(32, torch.rand(32, 3, 16, 16))
    assert z.shape == (32, 8) and torch.allclose(z.norm(dim=-1), torch.full((32,), math.sqrt(8)), atol=1e-4)
    phi = model.encode(torch.rand(2, 3, 3, 16, 16))
    assert phi.shape == (2, 3, 16) and torch.allclose(phi.norm(dim=-1), torch.full((2, 3), 4.0), atol=1e-4)
    assert model.encode_task(torch.rand(2, 3, 3, 16, 16)).shape == (2, 3, 8)


def test_infer_task_recovers_a_linear_reward():
    model = tdjepa().eval()
    model.psi_encoder = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, 8))  # well conditioned
    obs = torch.rand(256, 3, 16, 16)
    w = torch.randn(8)
    rewards = model.psi(obs).detach() @ w
    assert torch.allclose(model.infer_task(obs, rewards, project=False), w, atol=1e-3)
    assert torch.allclose(model.infer_task(obs, rewards), math.sqrt(8) * w / w.norm(), atol=1e-3)
    goals = model.goal_task(obs[:3], obs)
    assert goals.shape == (3, 8)


def test_policy_and_zero_shot_goal_reaching():
    model = tdjepa().eval()
    z, task = model.encode(torch.rand(4, 1, 3, 16, 16))[:, 0], model.sample_tasks(4)
    a = model.policy(z, task)
    assert a.shape == (4, 2) and a.abs().max() <= 1
    assert torch.equal(a, model.policy(z, task)) and not torch.equal(a, model.policy(z, task, explore=True))
    assert model.value(z, a, task).shape == (4,)
    env = TwoRooms(batch_size=4, size=16)
    start, goal = env.tasks(generator=torch.Generator().manual_seed(0))
    policies = {"TD-JEPA": model.controller(model.goal_task(env.render(goal))), "random": evaluation.random_policy(env)}
    result = evaluation.goal_reaching(env, policies, start, goal, steps=3)
    assert result.paths["TD-JEPA"].shape == (4, 4, 2)


def test_update_target_moves_toward_online():
    model = tdjepa()
    with torch.no_grad():
        for p in model.phi_predictor.parameters():
            p.add_(1.0)
    before = [p.clone() for p in model.targets["phi_predictor"].parameters()]
    model.update_target(0.5)
    after = list(model.targets["phi_predictor"].parameters())
    online = list(model.phi_predictor.parameters())
    assert all(torch.allclose(b2, 0.5 * (b1 + o)) for b1, b2, o in zip(before, after, online, strict=True))
    assert len(model.param_groups(1e-4, 1e-5)) == 2
