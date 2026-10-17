import os

import pytest
import torch

from torch_world.benchmarks.lewam_arm import make_model
from torch_world.benchmarks.lewam_goal_reaching import goal_reaching_config
from torch_world.benchmarks.lewam_matched import epoch_indices, lr_scale
from torch_world.benchmarks.lewam_reference import ReferenceLeWAM, initialize_local_from_reference


def test_epoch_visits_each_valid_start_once_and_schedule_matches_reference():
    ids, starts = epoch_indices(3, 20, 5, 42)
    assert sorted(zip(ids.tolist(), starts.tolist())) == [(i, t) for i in range(3) for t in range(16)]
    other = epoch_indices(3, 20, 5, 43)
    assert not torch.equal(starts, other[1])
    assert lr_scale(0, 50) == 0.2
    assert lr_scale(4, 50) == 1
    assert lr_scale(5, 50) == 1
    assert 0 < lr_scale(49, 50) < 0.01


def test_full_reference_adapter_identical_initial_predictions_and_differentiable_loss():
    root = os.getenv("LEWAM_REFERENCE")
    if not root:
        pytest.skip("Requires explicitly supplied reference checkout")
    pytest.importorskip("torchvision")
    c = goal_reaching_config()
    ref = ReferenceLeWAM(c, root, torch.tensor([0.1, -0.2]), torch.tensor([0.3, 0.2]))
    local = make_model(c, action_mean=ref.action_mean, action_std=ref.action_std)
    initialize_local_from_reference(local, ref)
    local.eval()
    ref.eval()
    pixels = torch.rand(2, 2, 3, 224, 224)
    actions = torch.randn(2, 10, 2)
    with torch.no_grad():
        z, rz = local.encode(pixels), ref.encode(pixels)
        torch.testing.assert_close(z, rz, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(
            local.predict_future(z, actions), ref.predict_future(rz, actions), atol=1e-6, rtol=1e-5
        )
        a = local.sample_actions(
            z, z[:, -1], remaining=torch.ones(2) * 0.5, generator=torch.Generator().manual_seed(42)
        )
        b = ref.sample_actions(
            rz, rz[:, -1], remaining=torch.ones(2) * 0.5, generator=torch.Generator().manual_seed(42)
        )
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    batch = {
        "history_pixels": pixels,
        "future_pixels": pixels[:, :1],
        "goal_pixels": pixels[:, -1],
        "actions": actions,
        "history_pad": torch.zeros(2, 2, dtype=torch.bool),
        "action_valid": torch.ones(2, 10, dtype=torch.bool),
        "state_valid": torch.ones(2, 1, dtype=torch.bool),
        "remaining": torch.ones(2) * 0.5,
    }
    ref.train()
    loss = ref.loss(batch)["loss"]
    loss.backward()
    assert torch.isfinite(loss) and ref.model.encoder.kp_conv.weight.grad.abs().sum() > 0


def test_ranking_audit_reports_regret_and_excludes_ties():
    from torch_world.benchmarks.lewam_prediction_audit import ranking_metrics

    actual = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    correct = ranking_metrics(actual, actual)
    assert correct["pairwise_agreement"] == [1.0, None]
    assert correct["selection_regret"] == [0.0, 0.0]
    reversed_rank = ranking_metrics(-actual, actual)
    assert reversed_rank["pairwise_agreement"][0] == 0
    assert reversed_rank["selection_regret"][0] == 2


def test_full_range_subgoals_preserve_alignment_and_reject_cross_episode_goals():
    from torch_world.benchmarks.lewam_goal_reaching import sample_subgoals

    data = {"states": torch.arange(2 * 101 * 4).reshape(2, 101, 4)}
    starts, initial, goals = sample_subgoals(data, [1, 2], start_indices=[0, 75])
    torch.testing.assert_close(initial, data["states"][torch.arange(2), starts])
    torch.testing.assert_close(goals, data["states"][torch.arange(2), starts + 25])
    with pytest.raises(ValueError, match="inside"):
        sample_subgoals(data, [1, 2], start_indices=[0, 76])


def test_diverse_collection_modes_are_reproducible_and_expert_can_hold():
    from torch_world.benchmarks.lewam_arm import demonstration_task, environment, expert

    config = goal_reaching_config(diverse_motion=True)
    pairs = [demonstration_task(seed, config) for seed in range(17, 25)]
    for seed, pair in zip(range(17, 25), pairs):
        for a, b in zip(pair, demonstration_task(seed, config)):
            torch.testing.assert_close(a, b)
    assert any(start[2:].abs().sum() > 0 for start, goal in pairs)
    with environment(config, batch_size=len(pairs), observation="state") as env:
        env.goal = torch.stack([p[1] for p in pairs])
        env.reset(torch.stack([p[0] for p in pairs]))
        held = torch.ones(len(pairs), dtype=torch.bool)
        for step in range(config.steps):
            env.step(expert(env, gain=config.expert_gain))
            if step >= config.steps - 10:
                held &= env.success(env.state, env.goal)
        assert held.all()


def test_reference_full_loss_and_encoder_gradients_match_with_coupled_noise(monkeypatch):
    root = os.getenv("LEWAM_REFERENCE")
    if not root:
        pytest.skip("Requires explicitly supplied reference checkout")
    pytest.importorskip("torchvision")
    from torch_world.benchmarks.lewam_arm import normalize_pixels, training_loss

    c = goal_reaching_config()
    ref = ReferenceLeWAM(c, root, torch.tensor([0.1, -0.2]), torch.tensor([0.3, 0.2]))
    local = make_model(c, action_mean=ref.action_mean, action_std=ref.action_std)
    initialize_local_from_reference(local, ref)
    # Deterministic center crops/dropout, while retaining gradients through all targets.
    ref.eval()
    local.eval()
    pixels = normalize_pixels(torch.randint(0, 256, (2, 4, 3, 224, 224)).float() / 255, c)
    batch = {
        "history_pixels": pixels[:, :2],
        "future_pixels": pixels[:, 2:3],
        "goal_pixels": pixels[:, 3],
        "actions": torch.rand(2, 10, 2) * 2 - 1,
        "history_pad": torch.tensor([[True, False], [False, False]]),
        "action_valid": torch.tensor([[True] * 3 + [False] * 7, [True] * 10]),
        "state_valid": torch.ones(2, 1, dtype=torch.bool),
        "remaining": torch.tensor([0.1, 0.7]),
    }
    noise = torch.randn(2, 10, 2)
    normal = torch.randn
    uniform = torch.rand
    # Implementations draw flow time/noise in different orders; couple those draws explicitly.
    monkeypatch.setattr(
        torch,
        "randn",
        lambda *shape, **kw: (
            noise.clone() if len(shape) == 1 and tuple(shape[0]) == (2, 10, 2) else normal(*shape, **kw)
        ),
    )
    monkeypatch.setattr(torch, "randn_like", lambda x: noise.to(x))
    monkeypatch.setattr(
        torch, "rand", lambda *shape, **kw: torch.full((2,), 0.4) if shape == (2,) else uniform(*shape, **kw)
    )
    torch.manual_seed(80)
    a = training_loss(local, batch)
    torch.manual_seed(80)
    b = ref.loss(batch)
    for key in a:
        torch.testing.assert_close(a[key], torch.as_tensor(b[key]), atol=2e-6, rtol=1e-5)
    a["loss"].backward()
    b["loss"].backward()
    torch.testing.assert_close(
        local.encoder.kp_conv.weight.grad, ref.model.encoder.kp_conv.weight.grad, atol=2e-5, rtol=1e-3
    )
    torch.testing.assert_close(
        local.blocks[0].qkv[0].weight.grad, ref.model.blocks[0].qkv_state.weight.grad, atol=2e-5, rtol=1e-3
    )


def test_reference_and_local_mixed_precision_backward_with_float_latent_boundary():
    root = os.getenv("LEWAM_REFERENCE")
    if not root:
        pytest.skip("Requires explicitly supplied reference checkout")
    pytest.importorskip("torchvision")
    from torch_world.benchmarks.lewam_arm import training_loss
    from torch_world.benchmarks.lewam_matched import float_latent_boundary

    c = goal_reaching_config()
    ref = ReferenceLeWAM(c, root, torch.zeros(2), torch.ones(2))
    local = make_model(c)
    initialize_local_from_reference(local, ref)
    float_latent_boundary({"local": local, "upstream": ref})
    batch = {
        "history_pixels": torch.rand(2, 2, 3, 224, 224),
        "future_pixels": torch.rand(2, 1, 3, 224, 224),
        "goal_pixels": torch.rand(2, 3, 224, 224),
        "actions": torch.rand(2, 10, 2),
        "history_pad": torch.zeros(2, 2, dtype=torch.bool),
        "action_valid": torch.ones(2, 10, dtype=torch.bool),
        "state_valid": torch.ones(2, 1, dtype=torch.bool),
        "remaining": torch.ones(2) * 0.5,
    }
    for model in (ref, local):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = ref.loss(batch)["loss"] if model is ref else training_loss(local, batch)["loss"]
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
