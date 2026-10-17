"""Reference protocol invariants and opt-in numerical parity with pinned upstream.

LEWAM_REFERENCE=/path/to/MinghaoFu/lewam pytest tests/models/test_lewam_reference.py
The external source is never downloaded or executed by ordinary test runs.
"""

import ast
import os
from pathlib import Path

import pytest
import torch
from torch import nn

from torch_world.models import LeWAM
from torch_world.models.jepa import LeWAMEncoder
from torch_world.models.jepa._lewam import _time_embedding


def small(**kwargs):
    return LeWAM(
        nn.Sequential(nn.Flatten(), nn.Linear(12, 8)),
        2,
        embed_dim=8,
        history=2,
        chunk_size=10,
        frameskip=5,
        num_states=1,
        dim=16,
        depth=2,
        num_heads=4,
        mlp_dim=64,
        projector_norm="batch",
        **kwargs,
    )


def test_spaced_targets_padding_and_physical_normalization():
    model = small(action_mean=[0.5, -0.2], action_std=[0.1, 0.3]).eval()
    raw = torch.tensor([[[0.9, -0.8]]])
    torch.testing.assert_close(model.denormalize_actions(model.normalize_actions(raw)), raw)
    # Four standard deviations must not be clipped to one before conversion.
    torch.testing.assert_close(model.denormalize_actions(torch.tensor([[[4.0, -2.0]]]), clamp=True), raw)
    z, clean, noisy = torch.randn(2, 2, 8), torch.randn(2, 10, 2), torch.randn(2, 10, 2)
    goal, time = torch.randn(2, 8), torch.rand(2)
    pad = torch.tensor([[True, False], [False, False]])
    for block in model.blocks:
        nn.init.normal_(block.modulation[-1].weight, std=0.05)
    state, action = model.joint_prediction(z, clean, noisy, time, goal, history_pad=pad)
    assert state.shape == (2, 1, 8)
    changed = z.clone()
    changed[0, 0] += 100
    check = model.joint_prediction(changed, clean, noisy, time, goal, history_pad=pad)
    torch.testing.assert_close(check[0], state)
    torch.testing.assert_close(check[1], action)
    changed = clean.clone()
    changed[:, 5:] += 100
    check = model.joint_prediction(z, changed, noisy, time, goal, history_pad=pad)
    torch.testing.assert_close(check[0], state)
    torch.testing.assert_close(check[1], action)
    with pytest.raises(ValueError, match="one-step"):
        model.predict(z, torch.randn(2, 2, 2))


def test_window_loss_joint_encoder_batch_and_validity():
    model = small()
    calls = []
    hook = model.projector.register_forward_hook(lambda m, args, out: calls.append(args[0].shape[0]))
    h, f, g = torch.randn(3, 2, 3, 2, 2), torch.randn(3, 1, 3, 2, 2), torch.randn(3, 3, 2, 2)
    f.requires_grad_()
    g.requires_grad_()
    losses = model.forward_windows(
        h, f, torch.randn(3, 10, 2), goal_pixels=g, action_valid=torch.zeros(3, 10), state_valid=torch.ones(3, 1)
    )
    assert calls == [12]  # H + S + goal, in one BatchNorm update.
    assert losses["action_loss"].item() == 0
    losses["loss"].backward()
    assert f.grad.abs().sum() > 0 and g.grad.abs().sum() > 0
    hook.remove()


def test_keypoint_encoder_shape_crop_and_eval_repeatability():
    encoder = LeWAMEncoder(crop_size=32).eval()
    x = torch.randn(2, 3, 40, 40)
    with torch.no_grad():
        a, b = encoder(x), encoder(x)
    assert a.shape == (2, 64) and a.abs().max() <= 1
    torch.testing.assert_close(a, b)
    with pytest.raises(ValueError, match="crop_size"):
        encoder(torch.randn(2, 3, 16, 16))


@pytest.mark.parametrize("goal_type", ["terminal", "sampled"])
def test_numerical_parity_with_upstream_mot(goal_type):
    location = os.getenv("LEWAM_REFERENCE")
    if not location:
        pytest.skip("Set LEWAM_REFERENCE to the audited upstream checkout")
    path = Path(location) / "lewam/models/lewam.py"
    tree = ast.parse(path.read_text())
    # Execute only the two audited classes; substitute only the unused encoder factory.
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": torch.nn.functional,
        "sinusoid": _time_embedding,
        "build_encoder": lambda *a, **k: nn.Identity(),
    }
    exec(compile(tree, str(path), "exec"), namespace)  # noqa: S102 - opt-in audited source parity
    cfg = {
        "fs": 5,
        "num_actions_pred": 10,
        "num_states_pred": 1,
        "action_raw_dim": 2,
        "history_len": 2,
        "goal_type": goal_type,
        "n_flow_steps": 8,
        "state_head": "mse",
        "action_head": "flow",
        "z_dim": 8,
        "d_model": 16,
        "n_heads": 4,
        "depth": 2,
        "dropout": 0.0,
        "encoder_backbone": "resnet18dp",
        "encoder_size": "tiny",
        "proj_hidden": 16,
        "img_size": 224,
    }
    upstream, ours = namespace["LeWAM"](cfg).eval(), small().eval()
    mapping = {
        "history_projection": "hist_proj",
        "goal_projection": "goal_proj",
        "action_projection": "action_proj",
        "time_projection": "tau_proj",
        "state_output": "state_out",
        "action_output": "action_out",
    }
    if goal_type == "sampled":
        mapping["horizon_projection"] = "h_proj"
    for local, remote in mapping.items():
        getattr(ours, local).load_state_dict(getattr(upstream, remote).state_dict())
    with torch.no_grad():
        ours.history_position.copy_(upstream.hist_pos_embed)
        ours.action_position.copy_(upstream.action_pos_embed)
        ours.state_queries.copy_(upstream.next_state_query)
        ours.state_position.copy_(upstream.next_state_pos)
        ours.token_type.weight.copy_(torch.cat([upstream.state_type.weight, upstream.action_type.weight]))
    for local, remote in zip(ours.blocks, upstream.blocks):
        nn.init.normal_(remote.ada_action[-1].weight, std=0.05)  # exercise nonzero gates
        local.modulation.load_state_dict(remote.ada_action.state_dict())
        for name in ("qkv", "out", "mlp"):
            for index, stream in enumerate(("state", "action")):
                getattr(local, name)[index].load_state_dict(getattr(remote, f"{name}_{stream}").state_dict())
    z = torch.randn(3, 2, 8, requires_grad=True)
    actions = torch.randn(3, 10, 2, requires_grad=True)
    noise, goal, time = torch.randn(3, 10, 2), torch.randn(3, 8), torch.rand(3)
    pad = torch.tensor([[True, False], [False, False], [True, False]])
    horizon = torch.rand(3) if goal_type == "sampled" else None
    future, velocity = ours.joint_prediction(z, actions, noise, time, goal, remaining=horizon, history_pad=pad)
    ref_velocity, ref_future = upstream.forward_mot(z, pad, actions[:, :5], noise, None, time, None, goal, horizon)
    torch.testing.assert_close(future, ref_future, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(velocity, ref_velocity, atol=1e-6, rtol=1e-5)
    grad = torch.autograd.grad(future.square().sum() + velocity.square().sum(), (z, actions), retain_graph=True)
    ref_grad = torch.autograd.grad(ref_future.square().sum() + ref_velocity.square().sum(), (z, actions))
    for a, b in zip(grad, ref_grad):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)


def test_numerical_parity_with_upstream_resnet_encoder():
    location = os.getenv("LEWAM_REFERENCE")
    if not location:
        pytest.skip("Set LEWAM_REFERENCE to the audited upstream checkout")
    pytest.importorskip("torchvision")
    namespace = {"torch": torch, "nn": nn}
    for filename, names in (
        ("module.py", {"MLP"}),
        ("encoders.py", {"Encoder", "batchnorm_to_groupnorm", "ResNetEncoder"}),
    ):
        path = Path(location) / "lewam/models" / filename
        tree = ast.parse(path.read_text())
        tree.body = [
            node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
        ]
        exec(compile(tree, str(path), "exec"), namespace)  # noqa: S102 - opt-in audited source parity
    upstream = namespace["ResNetEncoder"](output_dim=8, proj_hidden=16).eval()
    ours = LeWAM(LeWAMEncoder(), 2, embed_dim=8, feature_dim=64, projector_norm="batch").eval()
    ours.encoder.trunk.load_state_dict(upstream.trunk.state_dict())
    ours.encoder.kp_conv.load_state_dict(upstream.kp_conv.state_dict())
    ours.projector.load_state_dict(upstream.projector.net.state_dict())
    pixels = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        torch.testing.assert_close(ours.encoder(pixels), upstream.features(pixels), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(ours.encode(pixels[:, None])[:, 0], upstream(pixels), atol=1e-6, rtol=1e-5)
    # The per-image training crop must also agree for an identical RNG state.
    ours.encoder.train()
    upstream.train()
    torch.manual_seed(73)
    cropped = ours.encoder.crop(pixels)
    torch.manual_seed(73)
    reference_crop = upstream.crop(pixels)
    torch.testing.assert_close(cropped, reference_crop)


def test_action_loss_weights_each_example_equally():
    model = small().eval()
    history = torch.randn(2, 2, 3, 2, 2)
    future, goal = torch.randn(2, 1, 3, 2, 2), torch.randn(2, 3, 2, 2)
    actions = torch.randn(2, 10, 2)
    valid = torch.tensor([[1.0] + [0.0] * 9, [1.0] * 10])
    noise = torch.randn(actions.shape, generator=torch.Generator().manual_seed(15))

    # Zero predictions isolate the reduction, independently of attention/encoder.
    def prediction(z, clean, noisy, time, goal, remaining, pad):
        return z.new_zeros(2, 1, 8), torch.zeros_like(clean)

    model.joint_prediction = prediction
    result = model.forward_windows(
        history, future, actions, goal_pixels=goal, action_valid=valid, generator=torch.Generator().manual_seed(15)
    )
    error = (actions - noise).square().mean(-1)
    expected = (error[0, 0] + error[1].mean()) / 2
    torch.testing.assert_close(result["action_loss"], expected)


def test_planner_preserves_padding_and_standardized_action_units():
    from torch_world.planning import LeWAMPlanner

    model = small(action_mean=[0.5, -0.2], action_std=[0.1, 0.3]).eval()
    for block in model.blocks:
        nn.init.normal_(block.modulation[-1].weight, std=0.05)
    z, goal = torch.randn(2, 2, 8), torch.randn(2, 8)
    pad = torch.tensor([[True, False], [True, False]])
    planner = LeWAMPlanner(num_samples=2, refinement_steps=1)
    a = planner(model, z, goal, history_pad=pad, generator=torch.Generator().manual_seed(41))
    altered = z.clone()
    altered[:, 0] += 100
    b = planner(model, altered, goal, history_pad=pad, generator=torch.Generator().manual_seed(41))
    torch.testing.assert_close(a, b)
    assert a.shape == (2, 10, 2) and a.abs().max() <= 1
    assert all(p.grad is None for p in model.parameters())
