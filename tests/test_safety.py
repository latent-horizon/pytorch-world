import torch
from torch import nn

from torch_world.models.base import WorldModel
from torch_world.models.jepa._pldm import PLDM, PLDMEncoder
from torch_world.planning import CEM
from torch_world.safety import FailureMargin, FilteredPolicy, SafetyFilter, SafetyValue, fit_safety_value, imagine_step


class DoubleIntegrator(WorldModel):
    """Latent ``(position, velocity)``; the action accelerates: braking from high speed takes many steps."""

    history = 1

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))

    def encode(self, pixels):
        return pixels.flatten(2)[..., :2]

    def predict(self, z, actions):
        p, v = z.unbind(-1)
        v = v + 0.1 * actions[..., 0]
        return torch.stack([p + 0.1 * v, v], -1)


class PositionMargin(FailureMargin):
    """Failure when the position is negative."""

    def margin(self, z):
        return (2 * z[..., 0]).tanh()


def test_margin_loss_components_and_gradients():
    margin = FailureMargin(8, hidden=(16,))
    z, failure = torch.randn(32, 8), torch.arange(32) % 4 == 0
    out = margin(z, failure)
    assert all(out[k].dim() == 0 for k in ["loss", "safe_loss", "failure_loss", "accuracy"])
    assert torch.allclose(out["loss"], out["safe_loss"] + out["failure_loss"])
    s = margin.score(z)
    assert torch.allclose(out["safe_loss"], (margin.delta - s[~failure]).relu().mean())
    assert torch.allclose(out["failure_loss"], (margin.delta + s[failure]).relu().mean())
    out["loss"].backward()
    assert margin.net[0].weight.grad.abs().sum() > 0
    assert margin.margin(z).abs().max() < 1 and margin(z, torch.zeros(32))["failure_loss"] == 0


def test_margin_learns_the_sign_and_flattens_latents():
    torch.manual_seed(0)
    margin = FailureMargin((2, 3), hidden=(32,))
    z = torch.randn(512, 2, 3)
    failure = z[:, 0, 0] < -0.5
    opt = torch.optim.Adam(margin.parameters(), lr=1e-2)
    for _ in range(300):
        loss = margin(z, failure)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert ((margin.margin(z) < 0) == failure).float().mean() > 0.95
    assert margin.margin(torch.randn(4, 5, 2, 3)).shape == (4, 5)


def test_value_backup_actor_and_targets():
    torch.manual_seed(0)
    value = SafetyValue(3, 2, hidden=(16, 16), gamma=0.9, bounds=(-2.0, 2.0))
    z, a, z_next, l = torch.randn(10, 3), torch.rand(10, 2), torch.randn(10, 3), torch.randn(10)
    future = value.q(z_next, value.action(z_next, target=True), target=True)
    assert torch.allclose(value.target(l, z_next), 0.1 * l + 0.9 * torch.minimum(l, future))
    assert torch.allclose(value.target(l, z_next, gamma=0.0), l)
    assert value.action(100 * torch.randn(50, 3)).abs().max() <= 2.0
    assert value.value(z).shape == (10,)
    # the actor loss moves the actor only, the critic loss the critic only
    value(z, a, z_next, l)["actor_loss"].backward()
    assert value.critic[0].weight.grad is None and value.actor[0].weight.grad.abs().sum() > 0
    value.zero_grad()
    value(z, a, z_next, l)["critic_loss"].backward()
    assert value.actor[0].weight.grad is None or value.actor[0].weight.grad.abs().sum() == 0
    assert not any(p.requires_grad for p in value.critic_target.parameters())
    # Polyak update with rate tau
    with torch.no_grad():
        value.critic[0].weight.add_(1.0)
    before = value.critic_target[0].weight.clone()
    value.update_target()
    expected = before + value.tau * (value.critic[0].weight - before)
    assert torch.allclose(value.critic_target[0].weight, expected)


def test_reachability_sees_beyond_the_margin():
    """A state that is safe now but too fast to brake has a negative value: the failure is inevitable."""
    torch.manual_seed(0)
    model, margin = DoubleIntegrator(), PositionMargin(2)
    starts = torch.stack([torch.rand(4096) * 2.5 - 0.5, torch.rand(4096) * 4 - 2], -1)
    value = SafetyValue(2, 1, hidden=(64, 64), gamma=0.95)
    log = fit_safety_value(
        value,
        model,
        margin,
        starts,
        iterations=2500,
        warmup=300,
        envs=64,
        horizon=25,
        batch_size=256,
        critic_lr=3e-3,
        actor_lr=1e-3,
        generator=torch.Generator().manual_seed(0),
    )
    assert len(log["critic"]) == 2800 and not any(p.requires_grad for p in model.parameters())
    probe = torch.tensor([[0.3, -2.0], [0.3, 1.0], [1.5, 0.0], [-0.3, 1.0]])
    v, l = value.value(probe), margin.margin(probe)
    assert l[0] > 0.5 and v[0] < 0  # safe now, doomed
    assert v[1] > 0 and v[2] > 0  # moving away / far away
    assert v[3] < 0  # already failed
    assert value.action(torch.tensor([[0.5, -0.5]]))[0, 0] > 0  # the safety policy brakes


class Fixed(SafetyValue):
    """Value = position, safety policy = full push to the right."""

    def value(self, z):
        return z[..., 0]

    def action(self, z, target=False):
        return torch.ones(len(z), 1)

    def q(self, z, action, target=False, params=None):
        return z[..., 0] + 0.1 * action[..., 0]


def test_filter_keeps_safe_actions_and_overrides_unsafe_ones():
    model, value = DoubleIntegrator(), Fixed(2, 1, hidden=(4,))
    z = torch.tensor([[0.5, 0.0], [0.5, 0.0], [0.01, 0.0]])
    task = torch.tensor([[-1.0], [0.3], [-1.0]])
    assert torch.allclose(imagine_step(model, z, task)[:, 0], torch.tensor([0.49, 0.503, 0.0]))
    for rule in ["value", "q"]:
        out, override = SafetyFilter(model, value, threshold=0.2, rule=rule)(z, task)
        assert override.tolist() == [False, False, True]
        assert torch.allclose(out, torch.tensor([[-1.0], [0.3], [1.0]]))
    # threshold 0: nothing is overridden; a high threshold overrides everything
    assert not SafetyFilter(model, value, threshold=-1.0)(z, task)[1].any()
    assert SafetyFilter(model, value, threshold=1.0)(z, task)[1].all()


def test_filtered_planner_on_a_pixel_world_model():
    model = PLDM(PLDMEncoder(16, latent_dim=32, channels=(8, 8)), action_dim=2, latent_dim=32, idm_dim=64).eval()
    value = SafetyValue(32, 2, hidden=(16,))
    planner = CEM(horizon=3, action_dim=2, num_samples=16, num_elites=4, iterations=2, bounds=(-1, 1))
    goal = model.encode(torch.rand(2, 1, 3, 16, 16))[:, 0]

    def plan(obs):
        return planner(model, model.encode(obs[:, None]), goal)[:, 0]

    policy = SafetyFilter(model, value, threshold=0.0).wrap(plan)
    assert isinstance(policy, FilteredPolicy)
    action = policy(torch.rand(2, 3, 16, 16))
    assert action.shape == (2, 2) and action.abs().max() <= 1
    assert len(policy.overrides) == 1 and policy.overrides[0].shape == (2,)
    policy.reset()
    assert policy.overrides == []
