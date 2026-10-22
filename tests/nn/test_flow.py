import pytest
import torch

from torch_world.nn import euler_sample, interpolate, sample_time, time_embedding, velocity_target


def test_interpolation_endpoints_and_velocity():
    noise, target = torch.randn(3, 4, 2), torch.randn(3, 4, 2)
    torch.testing.assert_close(interpolate(noise, target, torch.zeros(3)), noise)
    torch.testing.assert_close(interpolate(noise, target, torch.ones(3)), target)
    time = torch.rand(3)
    x = interpolate(noise, target, time)
    torch.testing.assert_close(x + (1 - time)[:, None, None] * velocity_target(noise, target), target)


def test_euler_integrates_a_straight_flow_exactly():
    noise, target = torch.randn(5, 3), torch.randn(5, 3)
    calls = []

    def velocity(x, t):
        calls.append(t)
        return target - noise

    torch.testing.assert_close(euler_sample(velocity, noise, 4), target)
    assert [t[0].item() for t in calls] == [0.0, 0.25, 0.5, 0.75] and calls[0].shape == (5,)
    with pytest.raises(ValueError):
        euler_sample(velocity, noise, 0)


def test_time_sampling_and_embedding():
    generator = torch.Generator().manual_seed(0)
    uniform = sample_time(20000, generator=generator)
    flash = sample_time((200, 100), generator=generator, alpha=7.0)
    assert 0 <= uniform.min() and uniform.max() <= 1 and abs(uniform.mean() - 0.5) < 0.01
    # DreamZero-Flash: 1 - Beta(7, 1) has mean 1 / 8.
    assert flash.shape == (200, 100) and abs(flash.mean() - 0.125) < 0.005
    torch.testing.assert_close(
        sample_time(4, generator=torch.Generator().manual_seed(1), alpha=7.0),
        sample_time(4, generator=torch.Generator().manual_seed(1), alpha=7.0),
    )
    assert time_embedding(torch.rand(2, 3), 8).shape == (2, 3, 8)
    logit = sample_time(20000, generator=torch.Generator().manual_seed(2), logit_normal=(0.0, 1.0))
    assert 0 < logit.min() and logit.max() < 1 and abs(logit.mean() - 0.5) < 0.01
    # Logit-normal puts less mass near the ends than uniform.
    assert ((logit < 0.05) | (logit > 0.95)).float().mean() < 0.05
    with pytest.raises(ValueError):
        sample_time(3, alpha=7.0, logit_normal=(0.0, 1.0))
