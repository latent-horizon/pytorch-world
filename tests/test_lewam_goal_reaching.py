import ast
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from torch_world.benchmarks.lewam_arm import ArmConfig, environment, task, windows
from torch_world.models import LeWAM
from torch_world.planning import LeWAMHorizonPlanner


def test_sampled_goal_frames_and_horizons_are_same_episode_future_targets():
    c = ArmConfig(
        reference=True, size=224, goal_mode="sampled", frameskip=5, history_stride=5, history=2, chunk_size=10, steps=30
    )
    times = torch.arange(31)
    pixels = times[None, :, None, None, None].expand(2, -1, 3, 2, 2).byte()
    data = {
        "pixels": pixels,
        "goal_pixels": pixels[:, -1],
        "actions": torch.zeros(2, 30, 2),
        "states": times[None, :, None].expand(2, -1, 4),
        "goal": torch.zeros(2, 4),
    }
    b = windows(data, c, 512, torch.Generator().manual_seed(4))
    goal_time = ((b["goal_pixels"][:, 0, 0, 0] * 0.229 + 0.485) * 255).round()
    torch.testing.assert_close(goal_time, b["state"][:, 0] + b["remaining"] * c.goal_horizon * c.frameskip)
    assert (goal_time <= 30).all() and (goal_time > b["state"][:, 0]).all()
    assert ((b["remaining"] >= 0.1) & (b["remaining"] <= 1)).all()
    with pytest.raises(ValueError, match="history_stride"):
        replace(c, history_stride=1)


def test_visible_target_preserves_physics_and_stays_fixed_in_goal_image():
    c = ArmConfig(size=64)
    with environment(c) as plain, environment(replace(c, goal_marker=True)) as marked:
        start, goal = task(17)
        for env in (plain, marked):
            env.goal = goal[None]
            env.reset(start[None])
        assert plain.model.ngeom == marked.model.ngeom
        assert not torch.equal(plain.render(), marked.render())
        marked.set_visual_goal(goal[None])
        a = marked.render()
        marked.goal = start[None]
        torch.testing.assert_close(a, marked.render())
        marked.render(goal[None])
        for _ in range(5):
            action = torch.tensor([[0.1, -0.2]])
            plain.step(action)
            marked.step(action)
            torch.testing.assert_close(plain.state, marked.state, rtol=0, atol=0)
        with pytest.raises(ValueError, match="visual goal"):
            marked.set_visual_goal(torch.zeros(4))


def model():
    return LeWAM(
        nn.Linear(3, 8),
        2,
        embed_dim=8,
        history=2,
        chunk_size=4,
        frameskip=2,
        num_states=1,
        dim=16,
        depth=2,
        num_heads=4,
        mlp_dim=32,
        goal_type="sampled",
        flow_steps=2,
        action_mean=[0.1, -0.2],
        action_std=[0.3, 0.2],
    ).eval()


def test_horizon_planner_is_bounded_nonmutating_and_preserves_best_cost():
    m = model()
    z, g = torch.randn(2, 2, 8), torch.randn(2, 8)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    planner = LeWAMHorizonPlanner(num_samples=3, horizon=4, refinement_steps=3)
    with torch.no_grad():
        actions = planner(
            m, z, g, steps_to_goal=torch.tensor([1.0, 3.0]), history_pad=torch.tensor([[True, False], [False, False]])
        )
    assert actions.shape == (2, 6, 2) and actions.abs().max() <= 1 and not actions.requires_grad
    assert (planner.last_diagnostics["best_cost"] <= planner.last_diagnostics["warm_cost"]).all()
    for k, v in m.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    assert all(p.grad is None for p in m.parameters())
    m.goal_type = "terminal"
    with pytest.raises(ValueError, match="sampled-goal"):
        planner(m, z, g, steps_to_goal=torch.ones(2))


def test_gradient_planner_numerical_parity_with_upstream():
    location = os.getenv("LEWAM_REFERENCE")
    if not location:
        pytest.skip("Set LEWAM_REFERENCE for audited upstream parity")
    path = Path(location) / "lewam/eval/planners.py"
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    namespace = {"torch": torch, "np": np, "LeWAMPolicy": object, "LeWAM": object}
    exec(compile(tree, str(path), "exec"), namespace)  # noqa: S102 - opt-in audited reference
    m = model()

    class Adapter:
        goal_cond = True
        num_views = 1
        state_head = "mse"
        action_raw_dim = 2

        def sample_action_noise(self, b, device, generator):
            return torch.randn(b, m.chunk_size, 2, device=device, generator=generator)

        def sample_state_noise(self, b, device, generator):
            return torch.randn(b, 1, m.embed_dim, device=device, generator=generator)

        def predict_actions(self, h, p, *, action_noise, z_goal, h_norm):
            return m.normalize_actions(
                m.sample_actions(h, z_goal, history_pad=p, remaining=h_norm, noise=action_noise, clamp=False)
            )

        def predict_states(self, h, p, a, **kw):
            return m.predict_future(
                h,
                m.denormalize_actions(torch.nn.functional.pad(a, (0, 0, 0, m.chunk_size - m.frameskip))),
                history_pad=p,
            )

    ref = namespace["LeWAMPlanner"].__new__(namespace["LeWAMPlanner"])
    ref.model = Adapter()
    ref.num_proposals = 3
    ref.frameskip = 2
    ref.rollout_steps = 3
    ref.H_max = 4
    ref.plan_goal_time = True
    ref.exec_actions_per_plan = 2
    ref.device = "cpu"
    ref.grad_steps = 3
    ref.grad_lr = 0.05
    ref.grad_clip = 10.0
    ref.grad_tr = 0.01
    ref.grad_action_clip = 3.0
    ref.grad_all_k = True
    generator = torch.Generator().manual_seed(9)
    ref._flow_generator = lambda device: generator
    z, g = torch.randn(2, 2, 8), torch.randn(2, 8)
    pad = torch.tensor([[True, False], [False, False]])
    left = torch.tensor([1.0, 3.0])
    expected = ref._gradient_plan(z, pad, g, left)
    # Upstream accumulates model gradients; our API intentionally does not.
    m.zero_grad(set_to_none=True)
    ours = LeWAMHorizonPlanner(num_samples=3, horizon=4, refinement_steps=3)
    actual = ours(m, z, g, history_pad=pad, steps_to_goal=left, generator=torch.Generator().manual_seed(9))
    torch.testing.assert_close(actual, m.denormalize_actions(expected, clamp=True), atol=2e-6, rtol=1e-5)
    assert all(p.grad is None for p in m.parameters())
