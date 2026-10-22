import pytest
import torch
from torch import nn

from torch_world.models import DreamZero
from torch_world.nn import ConvVideoAE
from torch_world.planning import CEM, DreamZeroPolicy, WAMPlanner

K, H, M = 2, 4, 3


def tiny(**kwargs):
    return DreamZero(
        4,
        (4, 4),
        2,
        patch_size=2,
        chunk_frames=K,
        chunk_actions=H,
        num_chunks=M,
        dim=24,
        depth=2,
        num_heads=3,
        flow_steps=3,
        **kwargs,
    )


def batch(b=2):
    return torch.randn(b, 1 + M * K, 4, 4, 4), torch.rand(b, M * H, 2) * 2 - 1


def open_gates(model):
    """Randomize the zero-initialized AdaLN gates so that no zero shortcut can make a test pass."""
    for module in [*(b.modulation[-1] for b in model.blocks), model.final_modulation[-1]]:
        nn.init.normal_(module.weight, std=0.1)
    return model


def test_joint_training_gradients_and_masking():
    torch.manual_seed(0)
    model = open_gates(tiny(state_dim=3))
    latents, actions = batch()
    states = torch.randn(2, M, 3)
    out = model(latents, actions, states, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(out["loss"], out["video_loss"] + out["action_loss"])
    out["loss"].backward()
    for module in (model.video_in, model.video_out, model.action_encoder, model.action_decoder, model.state_encoder):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    masked = model(latents, actions, states, action_valid=torch.zeros(2, M * H))
    assert masked["action_loss"].item() == 0 and masked["video_loss"] > 0
    with pytest.raises(ValueError, match="state per chunk"):
        model(latents, actions)
    flash = tiny(flash_alpha=7.0)(latents, actions)
    assert torch.isfinite(flash["loss"])


def teacher_forced(model, latents, actions, noisy_video, noisy_actions, t):
    return model.joint_velocity(
        latents[:, 0],
        latents[:, 1 : 1 + (M - 1) * K],
        actions[:, : (M - 1) * H],
        noisy_video,
        noisy_actions,
        t,
        t,
        start=1,
    )


def test_block_causal_attention():
    torch.manual_seed(1)
    model = open_gates(tiny()).eval()
    latents, actions = batch()
    noisy_video, noisy_actions, t = torch.randn(2, M * K, 4, 4, 4), torch.randn(2, M * H, 2), torch.rand(2, M)
    video, action = teacher_forced(model, latents, actions, noisy_video, noisy_actions, t)

    def chunks(v, a, k):
        return v[:, : k * K], a[:, : k * H]

    # A later noisy chunk and its time never reach earlier chunks.
    later_video, later_actions, later_t = noisy_video.clone(), noisy_actions.clone(), t.clone()
    later_video[:, -K:] += 5
    later_actions[:, -H:] += 5
    later_t[:, -1] = 0.9 - later_t[:, -1]
    v, a = teacher_forced(model, latents, actions, later_video, later_actions, later_t)
    for x, y in zip(chunks(video, action, M - 1), chunks(v, a, M - 1)):
        torch.testing.assert_close(x, y)
    assert not torch.allclose(video[:, -K:], v[:, -K:])

    # Clean chunk 2 conditions noisy chunk 3 only; noisy chunk 2 must not see its own clean target.
    changed_latents, changed_actions = latents.clone(), actions.clone()
    changed_latents[:, 1 + K : 1 + 2 * K] += 5
    changed_actions[:, H : 2 * H] += 5
    v, a = teacher_forced(model, changed_latents, changed_actions, noisy_video, noisy_actions, t)
    for x, y in zip(chunks(video, action, 2), chunks(v, a, 2)):
        torch.testing.assert_close(x, y)
    assert not torch.allclose(action[:, -H:], a[:, -H:])


def test_teacher_forcing_matches_autoregressive_inference():
    torch.manual_seed(2)
    model = open_gates(tiny(state_dim=3)).eval()
    latents, actions = batch()
    states = torch.randn(2, M, 3)
    noisy_video, noisy_actions, t = torch.randn(2, M * K, 4, 4, 4), torch.randn(2, M * H, 2), torch.rand(2, M)
    video, action = model.joint_velocity(
        latents[:, 0],
        latents[:, 1 : 1 + (M - 1) * K],
        actions[:, : (M - 1) * H],
        noisy_video,
        noisy_actions,
        t,
        t,
        start=1,
        states=states,
    )
    for k in range(1, M + 1):
        j = k - 1
        v, a = model.joint_velocity(
            latents[:, 0],
            latents[:, 1 : 1 + j * K],
            actions[:, : j * H],
            noisy_video[:, j * K : k * K],
            noisy_actions[:, j * H : k * H],
            t[:, j:k],
            t[:, j:k],
            start=k,
            states=states[:, :k],
        )
        torch.testing.assert_close(v, video[:, j * K : k * K], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(a, action[:, j * H : k * H], atol=1e-5, rtol=1e-4)


def test_sampling_dynamics_and_context_window():
    torch.manual_seed(3)
    model = open_gates(tiny(action_scale=[1.0, 0.2])).eval()
    latents, actions = batch()
    noise, video_noise = torch.randn(2, H, 2), torch.randn(2, K, 4, 4, 4)
    future, chunk = model.imagine(
        latents[:, : 1 + K], past_actions=actions[:, :H], noise=noise, video_noise=video_noise
    )
    assert future.shape == (2, K, 4, 4, 4) and chunk.shape == (2, H, 2)
    assert chunk[..., 1].abs().max() <= 0.2
    same = model.sample_actions(latents[:, : 1 + K], past_actions=actions[:, :H], noise=noise, video_noise=video_noise)
    torch.testing.assert_close(same, chunk)
    # Context beyond num_chunks - 1 chunks is dropped and the frame before the window becomes the anchor.
    long = model.imagine(latents, past_actions=actions, noise=noise, video_noise=video_noise)
    short = model.imagine(latents[:, K:], past_actions=actions[:, H:], noise=noise, video_noise=video_noise)
    torch.testing.assert_close(long[0], short[0])
    torch.testing.assert_close(long[1], short[1])

    plan = actions[:, :H].clone().requires_grad_(True)
    predicted = model.predict_future(latents[:, :1], plan, video_noise=video_noise)
    assert predicted.shape == (2, K, 4, 4, 4)
    predicted.square().sum().backward()
    assert plan.grad.abs().sum() > 0
    assert model.rollout(latents[:, :1], actions[:, : 2 * H]).shape == (2, 1 + 2 * K, 4, 4, 4)
    plan = CEM(2 * H, 2, num_samples=6, num_elites=2, iterations=1)(model, latents[:, :1], latents[:, -1])
    assert plan.shape == (2, 2 * H, 2)
    with pytest.raises(ValueError, match="history_pad"):
        model.sample_actions(latents[:, :1], history_pad=torch.zeros(2, 1, dtype=torch.bool))


def test_goal_conditioned_planner():
    torch.manual_seed(4)
    model = open_gates(tiny(goal_conditioned=True)).eval()
    latents, actions = batch()
    goal = latents[:, -1]
    with pytest.raises(ValueError, match="goal"):
        model.sample_actions(latents[:, :1])
    plan = WAMPlanner(num_samples=3, refinement_steps=1)(
        model, latents[:, : 1 + K], goal, actions[:, :H], generator=torch.Generator().manual_seed(0)
    )
    assert plan.shape == (2, H, 2) and torch.isfinite(plan).all()
    assert WAMPlanner()(model, latents[:, :1], goal).shape == (2, H, 2)


def test_closed_loop_policy_uses_real_observations():
    torch.manual_seed(5)
    autoencoder = ConvVideoAE(latent_channels=4, width=8, depth=2)
    model = open_gates(tiny(autoencoder=autoencoder)).eval()
    encoded = []
    encode = autoencoder.encode
    autoencoder.encode = lambda pixels: encoded.append(pixels.shape[1]) or encode(pixels)
    policy = DreamZeroPolicy(model, generator=torch.Generator().manual_seed(0))
    assert policy.stride == 2
    actions = [policy(torch.rand(2, 3, 16, 16)) for _ in range(4 * H)]
    assert all(a.shape == (2, 2) for a in actions)
    # One plan per chunk, each from the real frames: the anchor, then K raw frames per past chunk, capped at M - 1.
    assert encoded == [1, 1 + K, 1 + 2 * K, 1 + 2 * K]
    assert model.decode(model.encode(torch.rand(1, 2, 3, 16, 16))).shape == (1, 2, 3, 16, 16)
    policy.reset()
    assert not policy.frames and not policy.actions


def test_kv_cache_matches_full_sequence():
    torch.manual_seed(6)
    model = open_gates(tiny(state_dim=3, goal_conditioned=True)).eval()
    latents, actions = batch()
    states, goal = torch.randn(2, M, 3), latents[:, -1]
    noisy_video, noisy_actions, t = torch.randn(2, K, 4, 4, 4), torch.randn(2, H, 2), torch.rand(2, 1)
    for j in range(M):
        context = (latents[:, 0], latents[:, 1 : 1 + j * K], actions[:, : j * H])
        full = model.joint_velocity(*context, noisy_video, noisy_actions, t, t, start=j + 1, states=states, goal=goal)
        cache = model.context_cache(*context, states=states, goal=goal)
        cached = model.cached_velocity(cache, noisy_video, noisy_actions, t, t, start=j + 1, states=states)
        for x, y in zip(full, cached):
            torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-4)


def test_flex_attention_matches_dense_mask():
    torch.manual_seed(7)
    dense = open_gates(tiny(goal_conditioned=True)).eval()
    flex = tiny(goal_conditioned=True, attention="flex").eval()
    flex.load_state_dict(dense.state_dict())
    latents, actions = batch()
    generator = lambda: torch.Generator().manual_seed(3)
    with torch.no_grad():  # FlexAttention has no CPU backward; it trains on CUDA.
        a = dense(latents, actions, goal=latents[:, -1], generator=generator())
        b = flex(latents, actions, goal=latents[:, -1], generator=generator())
    torch.testing.assert_close(a["loss"], b["loss"], atol=1e-5, rtol=1e-4)
    assert dense.attention_mask(M - 1, M, 1) is dense.attention_mask(M - 1, M, 1)
    with pytest.raises(ValueError, match="attention"):
        tiny(attention="xformers")


def test_inverse_dynamics_and_loss_options():
    torch.manual_seed(8)
    model = open_gates(tiny()).eval()
    latents, actions = batch()
    noise = torch.randn(2, H, 2)
    future = latents[:, 1 + K : 1 + 2 * K]
    inferred = model.inverse_dynamics(latents[:, : 1 + K], future, past_actions=actions[:, :H], noise=noise)
    assert inferred.shape == (2, H, 2) and inferred.abs().max() <= 1
    other = model.inverse_dynamics(latents[:, : 1 + K], future + 3, past_actions=actions[:, :H], noise=noise)
    assert not torch.allclose(inferred, other)
    with pytest.raises(ValueError, match="one chunk"):
        model.inverse_dynamics(latents[:, :1], future[:, :1])

    policy_only = tiny(video_weight=0.0, time_schedule="logit_normal")
    out = policy_only(latents, actions, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(out["loss"], out["action_loss"])
    assert out["video_loss"] > 0
    with pytest.raises(ValueError, match="time_schedule"):
        tiny(time_schedule="cosine")
