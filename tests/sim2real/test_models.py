from dataclasses import replace

import pytest
import torch

from torch_world.planning import CEM
from torch_world.sim2real import (
    ChessTaskConfig,
    Episode,
    EpisodeWindows,
    Observation,
    SO101Config,
    WorldModelPolicy,
    load_checkpoint,
    make_adapter,
    save_checkpoint,
)


@pytest.mark.parametrize("kind", ["lewm", "dinowm", "robojepa"])
def test_training_history_planning_and_checkpoint(kind, tmp_path):
    torch.manual_seed(3)
    adapter = make_adapter(kind, size=16, smoke=True)
    pixels, actions, states = torch.rand(2, 4, 3, 16, 16), torch.rand(2, 4, 4), torch.rand(2, 4, 6)
    loss = adapter.loss(pixels, actions, states)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in adapter.parameters())
    obs = Observation(pixels[0, 0], states[0, 0, :5], 0.5, 0.0)
    policy = WorldModelPolicy(
        adapter, obs, planner=CEM(3, 4, num_samples=8, num_elites=2, iterations=1, bounds=(-1, 1))
    )
    for _ in range(5):
        action = policy(obs)
        assert action.shape == (4,) and torch.isfinite(action).all() and (action.abs() <= 1).all()
        with pytest.raises(RuntimeError, match="record_action"):
            policy(obs)
        policy.record_action(action * 0.5)
    assert len(policy.observations) == 3 and len(policy.actions) == 2
    config, task = SO101Config(size=16), ChessTaskConfig()
    path = tmp_path / "model.pt"
    save_checkpoint(path, adapter, config, task)
    loaded = load_checkpoint(path, config, task)
    torch.testing.assert_close(adapter.encode(pixels, states), loaded.encode(pixels, states))
    with pytest.raises(ValueError, match="contract"):
        load_checkpoint(path, replace(config, max_step=0.001), task)


def test_windows_never_cross_episode_boundaries():
    episodes = [
        Episode(
            torch.full((5, 3, 16, 16), float(i)),
            torch.zeros(5, 6),
            torch.arange(16).reshape(4, 4).float(),
            torch.zeros(4, 4),
            {"seed": i},
        )
        for i in (0, 1)
    ]
    windows = EpisodeWindows(episodes, 3)
    assert len(windows) == 4
    for index in range(len(windows)):
        window = windows[index]
        assert window["pixels"].unique().numel() == 1
        assert window["actions"].shape == (4, 4)
        assert window["actions"][-1].eq(0).all()
    torch.testing.assert_close(windows[1]["actions"][:3], episodes[0].actions[1:])


@pytest.mark.parametrize("delay", [0, 1, 2])
def test_windows_align_delayed_pixels_states_and_cached_features(delay):
    # A step returns frame max(0, step - delay), but records the action
    # actually applied at that step. Only frames with successors are usable.
    frames = (torch.arange(7) - delay).clamp_min(0).float()
    episode = Episode(
        frames[:, None, None, None].expand(-1, 3, 2, 2),
        frames[:, None].expand(-1, 6),
        torch.arange(6).float()[:, None].expand(-1, 4),
        torch.zeros(6, 4),
        {"randomization": {"observation_delay": delay}},
    )
    windows = EpisodeWindows([episode], 2)
    cached = EpisodeWindows([episode], 2, [frames[:, None]])
    assert len(windows) == 5 - delay
    for t in range(len(windows)):
        expected = torch.arange(t, t + 3).float()
        torch.testing.assert_close(windows[t]["pixels"][:, 0, 0, 0], expected)
        torch.testing.assert_close(windows[t]["states"][:, 0], expected)
        torch.testing.assert_close(cached[t]["features"][:, 0], expected)
        torch.testing.assert_close(windows[t]["actions"][:2, 0], expected[:2])


def test_lewm_measured_state_and_multistep_checkpoint(tmp_path):
    torch.manual_seed(31)
    adapter = make_adapter("lewm", size=16, smoke=True, proprioception=True, rollout_steps=8)
    assert adapter.history == 3 and adapter.training_window == 10
    pixels, actions, states = torch.rand(2, 11, 3, 16, 16), torch.rand(2, 11, 4), torch.rand(2, 11, 6)
    # adaLN-zero intentionally starts with zero action gates. Exercise the
    # gradients after optimization has opened those gates.
    optimizer = torch.optim.Adam(adapter.parameters(), lr=0.003)
    for _ in range(3):
        optimizer.zero_grad()
        adapter.loss(pixels, actions, states)["loss"].backward()
        optimizer.step()
    optimizer.zero_grad()
    adapter.eval()
    z = adapter.encode(pixels, states)
    torch.testing.assert_close(z[..., -6:], adapter.proprio(states))
    changed = adapter.encode(pixels, states + 0.2)
    torch.testing.assert_close(changed[..., :-6], z[..., :-6])
    assert not torch.allclose(changed[..., -6:], z[..., -6:])
    out = adapter.loss(pixels, actions, states)
    assert all(torch.isfinite(value) for value in out.values())
    out["rollout_loss"].backward()
    assert adapter.model.action_encoder[0].weight.grad.abs().sum() > 0
    assert adapter.model.encoder[0].weight.grad.abs().sum() > 0
    # A later action affects autoregressive training but not the first three
    # teacher-forced predictions. The unused final action affects neither.
    later = actions.clone()
    later[:, 6] += 10
    altered = adapter.loss(pixels, later, states)
    torch.testing.assert_close(out["pred_loss"], altered["pred_loss"])
    assert not torch.allclose(out["rollout_loss"], altered["rollout_loss"])
    padded = actions.clone()
    padded[:, -1] += 100
    torch.testing.assert_close(out["rollout_loss"], adapter.loss(pixels, padded, states)["rollout_loss"])
    # Future state targets must never be inputs to autoregressive predictions.
    future = z.detach().clone()
    future[:, 3:, -6:] += 100
    a = adapter.model.rollout(z[:, :3], actions[:, :10])
    b = adapter.model.rollout(future[:, :3], actions[:, :10])
    torch.testing.assert_close(a, b)
    with pytest.raises(ValueError, match="Not enough"):
        adapter.loss(pixels[:, :4], actions[:, :4], states[:, :4])
    config, task = SO101Config(size=16), ChessTaskConfig()
    path = tmp_path / "state-rollout.pt"
    save_checkpoint(path, adapter, config, task)
    restored = load_checkpoint(path, config, task)
    assert restored.recipe == adapter.recipe and restored.training_window == 10
    torch.testing.assert_close(restored.encode(pixels, states), z)


def test_lewm_long_windows_remain_episode_bounded():
    adapter = make_adapter("lewm", size=16, proprioception=True, rollout_steps=8)
    episodes = [
        Episode(
            torch.full((12, 3, 16, 16), float(i)),
            torch.full((12, 6), float(i)),
            torch.ones(11, 4),
            torch.ones(11, 4),
            {},
        )
        for i in (0, 1)
    ]
    windows = EpisodeWindows(episodes, adapter.training_window)
    assert len(windows) == 4
    for w in windows:
        assert len(w["pixels"]) == 11 and w["pixels"].unique().numel() == 1
        assert len(w["actions"]) == 11 and w["actions"][-1].eq(0).all()
