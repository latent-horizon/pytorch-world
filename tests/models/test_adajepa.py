import torch
from torch import nn

from torch_world.envs import TwoRooms
from torch_world.evaluation import goal_reaching
from torch_world.models.jepa._adajepa import AdaJEPA
from torch_world.planning import CEM


def adajepa(**kwargs):
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 32), nn.GELU(), nn.Linear(32, 32))
    return AdaJEPA(
        encoder, action_dim=2, embed_dim=32, action_embed_dim=8, history=3, depth=2, num_heads=2, head_dim=8,
        mlp_dim=64, **kwargs,
    )  # fmt: skip


def test_loss_components_and_gradients():
    model = adajepa()
    out = model(torch.randn(4, 4, 3, 8, 8), torch.randn(4, 4, 2))
    assert all(torch.isfinite(v) for v in out.values())
    total = out["pred_loss"] + 0.1 * out["curvature_loss"] + out["std_loss"] + 0.04 * out["cov_loss"]
    assert torch.allclose(out["loss"], total)
    out["pred_loss"].backward()
    # stop-gradient targets: the encoder still learns through the context latents
    assert model.encoder[1].weight.grad.abs().sum() > 0
    assert model.action_encoder.weight.grad.abs().sum() > 0


def test_prediction_is_causal_and_shaped():
    model = adajepa().eval()
    z, a = torch.randn(2, 3, 32), torch.randn(2, 3, 2)
    a2 = a.clone()
    a2[:, 2] = torch.randn(2, 2)
    with torch.no_grad():
        p, p2 = model.predict(z, a), model.predict(z, a2)
        out = model.rollout(z[:, :1], torch.randn(2, 5, 2))
    assert p.shape == (2, 3, 32) and torch.allclose(p[:, :2], p2[:, :2], atol=1e-6)
    assert not torch.allclose(p[:, 2], p2[:, 2])
    assert out.shape == (2, 6, 32) and torch.equal(out[:, :1], z[:, :1])


def test_adapt_updates_only_the_adapted_layers_and_lowers_the_loss():
    torch.manual_seed(0)
    model = adajepa().eval()
    pixels, actions = torch.randn(1, 6, 3, 8, 8), torch.randn(1, 5, 2)
    predictor, head = model.adapted_parameters()
    assert head == list(model.encoder[3].parameters())
    adapted = {id(p) for p in predictor + head}
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    losses = model.adapt(pixels, actions, steps=20, lr=1e-2, encoder_lr=1e-3)
    assert losses[-1] < losses[0]
    for n, p in model.named_parameters():
        assert torch.equal(p, before[n]) != (id(p) in adapted), n
    assert all(p.requires_grad for p in model.parameters())
    frozen = adajepa().eval()
    head_before = [p.clone() for p in frozen.adapted_parameters()[1]]
    frozen.adapt(pixels, actions, steps=1, encoder_lr=0)
    assert all(torch.equal(a, b) for a, b in zip(frozen.adapted_parameters()[1], head_before))


def test_policy_adapts_per_episode_and_restores_weights():
    torch.manual_seed(0)
    model = adajepa().eval()
    env = TwoRooms(batch_size=2, size=8)
    start, goal = env.tasks(generator=torch.Generator().manual_seed(0))
    planner = CEM(horizon=3, action_dim=2, num_samples=8, num_elites=2, iterations=2, bounds=(-1, 1))
    weights = {n: p.detach().clone() for n, p in model.named_parameters()}
    policy = model.policy(planner, env.render(goal), buffer=2, lr=1e-2)
    run = goal_reaching(env, {"ada": policy}, start, goal, steps=4)
    assert run.paths["ada"].shape == (2, 5, 2)
    assert len(policy.losses) == 3 and all(len(step) == 2 for step in policy.losses)
    assert all(torch.equal(p, weights[n]) for n, p in model.named_parameters())
    assert not all(torch.equal(a, b) for a, b in zip(policy.state[0], policy.initial))
    assert not all(torch.equal(a, b) for a, b in zip(policy.state[0], policy.state[1]))
    assert len(policy.frames[0]) == 3 and len(policy.actions[0]) == 3  # two buffered transitions + the pending action
    policy.reset()
    assert not policy.state and not policy.frames
