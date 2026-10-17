import pytest
import torch
from torch import nn

from torch_world.losses import SIGReg
from torch_world.models import LeWAM
from torch_world.planning import CEM, LeWAMPlanner


def tiny(**kwargs):
    return LeWAM(
        nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 16)),
        2,
        16,
        history=2,
        chunk_size=3,
        dim=24,
        depth=2,
        num_heads=3,
        mlp_dim=48,
        flow_steps=3,
        sigreg=SIGReg(16),
        **kwargs,
    )


def batch():
    return torch.rand(4, 5, 3, 8, 8), torch.rand(4, 5, 2) * 2 - 1


def test_joint_training_gradients_and_padding():
    torch.manual_seed(5)
    model = tiny()
    pixels, actions = batch()
    goal = torch.rand(4, 3, 8, 8)
    out = model(pixels, actions, goal_pixels=goal, generator=torch.Generator().manual_seed(2))
    torch.testing.assert_close(out["loss"], out["pred_loss"] + out["action_loss"] + 0.04 * out["sigreg_loss"])
    out["loss"].backward()
    for module in (model.encoder, model.state_output, model.action_output):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    actions[:, -1] += 1000
    other = model(pixels, actions, goal_pixels=goal, generator=torch.Generator().manual_seed(2))
    for key in out:
        torch.testing.assert_close(out[key], other[key])
    masked = model(pixels, actions, goal_pixels=goal, action_valid=torch.zeros(4, 3))
    assert masked["action_loss"].item() == 0 and masked["pred_loss"] > 0
    with pytest.raises(ValueError, match="explicit goal_pixels"):
        model(pixels, actions)


def test_attention_prevents_target_leakage_after_training():
    torch.manual_seed(8)
    model = tiny().eval()
    # Open AdaLN gates so a zero-init shortcut cannot make this test pass.
    for block in model.blocks:
        nn.init.normal_(block.modulation[-1].weight, std=0.1)
    z = torch.randn(2, 2, 16)
    clean, noisy, goal, time = torch.randn(2, 3, 2), torch.randn(2, 3, 2), torch.randn(2, 16), torch.rand(2)
    future, velocity = model.joint_prediction(z, clean, noisy, time, goal)
    changed_future, same_velocity = model.joint_prediction(z, clean + 7, noisy, time, goal)
    torch.testing.assert_close(velocity, same_velocity)
    assert not torch.allclose(future, changed_future)
    same_future, changed_velocity = model.joint_prediction(z, clean, noisy + 7, time + 0.4, goal + 4)
    torch.testing.assert_close(future, same_future)
    assert not torch.allclose(velocity, changed_velocity)
    _, goal_velocity = model.joint_prediction(z, clean, noisy, time, goal + 4)
    assert not torch.allclose(velocity, goal_velocity)
    later = clean.clone()
    later[:, 2] += 20
    a, _ = model.joint_prediction(z, later, noisy, time, goal)
    torch.testing.assert_close(future[:, :2], a[:, :2])
    later_noise = noisy.clone()
    later_noise[:, 2] += 20
    _, v = model.joint_prediction(z, clean, later_noise, time, goal)
    torch.testing.assert_close(velocity[:, :2], v[:, :2])


def test_state_residual_rollout_sampling_and_checkpoint(tmp_path):
    model = tiny(state_dim=2, action_scale=[1, 0.15]).eval()
    pixels, actions = batch()
    states = torch.randn(4, 5, 2)
    z = model.encode(pixels, states)
    future = model.predict_future(z[:, :2], actions[:, 1:4])
    torch.testing.assert_close(future[..., -2:], states[:, 1:2].expand(-1, 3, -1))
    kwargs = {"goal_pixels": pixels[:, -1], "goal_states": states[:, -1]}
    model(pixels, actions, states, **kwargs)["loss"].backward()
    assert model.state_output[-1].weight.grad[-2:].abs().sum() > 0
    a = model.sample_actions(z[:, :2], z[:, -1], generator=torch.Generator().manual_seed(3))
    b = model.sample_actions(z[:, :2], z[:, -1], generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(a, b)
    assert a.shape == (4, 3, 2) and (a.abs() <= model.action_scale).all()
    rollout = model.rollout(z[:, :2], actions[:, :4])
    assert rollout.shape == (4, 5, 18)
    torch.testing.assert_close(rollout[:, :2], z[:, :2])
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    restored = tiny(state_dim=2, action_scale=[1, 0.15]).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(restored.encode(pixels, states), z)
    model.zero_grad(set_to_none=True)
    plan = LeWAMPlanner(num_samples=3, refinement_steps=2)(model, z[:, :2], z[:, -1])
    assert plan.shape == a.shape and torch.isfinite(plan).all()
    assert all(p.grad is None for p in model.parameters())
    plan = CEM(2, 2, num_samples=4, num_elites=2, iterations=1)(model, z[:, :2], z[:, -1], past_actions=actions[:, :1])
    assert plan.shape == (4, 2, 2)


def test_goal_free_variant_and_invalid_configuration():
    model = tiny(goal_conditioned=False)
    pixels, actions = batch()
    assert torch.isfinite(model(pixels, actions)["loss"])
    assert model.sample_actions(model.encode(pixels[:, :2])).shape == (4, 3, 2)
    with pytest.raises(ValueError, match="action_scale"):
        tiny(action_scale=[1, 0])


def test_planner_sampling_controls_preserve_default_and_determinism():
    model = tiny().eval()
    pixels, _ = batch()
    z = model.encode(pixels[:, :2])
    goal = z[:, -1]
    expected = model.sample_actions(z, goal, generator=torch.Generator().manual_seed(17))
    actual = LeWAMPlanner()(model, z, goal, generator=torch.Generator().manual_seed(17))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    deterministic = LeWAMPlanner(noise_scale=0, flow_steps=5)
    a = deterministic(model, z, goal, generator=torch.Generator().manual_seed(1))
    b = deterministic(model, z, goal, generator=torch.Generator().manual_seed(99))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    expected = model.sample_actions(z, goal, noise=z.new_zeros(4, 3, 2), steps=5)
    torch.testing.assert_close(a, expected, rtol=0, atol=0)
    assert not a.requires_grad and (a.abs() <= model.action_scale).all()
    for kwargs in ({"noise_scale": -1}, {"noise_scale": float("nan")}, {"flow_steps": 0}, {"flow_steps": 1.5}):
        with pytest.raises(ValueError):
            LeWAMPlanner(**kwargs)
