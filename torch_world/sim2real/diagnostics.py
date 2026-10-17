"""Held-out diagnostics for the SO101 LeWAM pipeline, without robot control."""

import torch

from torch_world.sim2real.data import LeWAMWindows


@torch.no_grad()
def diagnose_lewam(adapter, episodes, *, windows_per_episode=4, seed=9017):
    """Compare aligned predictions/actions against persistence and zero actions.

    Sample evenly spaced windows within each held-out episode. State errors are
    measured against aligned observations; they are not directly comparable to
    earlier reports using uncorrected observation delay. Action errors include
    only successful demonstrations and use per-coordinate action-scale units.
    No held-out examples update parameters or select checkpoints.
    """
    if adapter.kind != "lewam" or windows_per_episode < 1:
        raise ValueError("Expected a LeWAM adapter and positive windows_per_episode")
    adapter.eval()
    model, h, k = adapter.model, adapter.history, adapter.model.chunk_size
    dataset = LeWAMWindows(episodes, h, k)
    groups = {}
    for i, (e, _) in enumerate(dataset.index):
        groups.setdefault(e, []).append(i)
    indices = [
        entries[i]
        for entries in groups.values()
        for i in torch.linspace(0, len(entries) - 1, min(windows_per_episode, len(entries))).round().long().tolist()
    ]
    batch = {key: torch.stack([dataset[i][key] for i in indices]).to(adapter.device) for key in dataset[indices[0]]}
    z = adapter.encode(batch["pixels"], batch["states"])
    goals = adapter.encode(batch["goal_pixels"][:, None], batch["goal_states"][:, None])[:, 0]
    actions = batch["actions"][:, h - 1 : h + k - 1]
    predicted = model.predict_future(z[:, :h], actions)
    target, persistence = z[:, h:], z[:, h - 1 : h].expand_as(predicted)
    zero = model.predict_future(z[:, :h], torch.zeros_like(actions))
    v = model.embed_dim
    result = {
        "windows": len(indices),
        "alignment": "constant observation delay corrected; applied actions",
        "one_step_visual_mse": float((predicted[:, 0, :v] - target[:, 0, :v]).square().mean()),
        "one_step_persistence_visual_mse": float((persistence[:, 0, :v] - target[:, 0, :v]).square().mean()),
        "chunk_visual_mse": float((predicted[..., :v] - target[..., :v]).square().mean()),
        "chunk_persistence_visual_mse": float((persistence[..., :v] - target[..., :v]).square().mean()),
        "chunk_zero_action_visual_mse": float((zero[..., :v] - target[..., :v]).square().mean()),
        "visual_embedding_mean_std": float(z[..., :v].flatten(0, 1).std(0).mean()),
    }
    if model.state_dim:
        for name, p in (("model", predicted), ("persistence", persistence)):
            result[f"{name}_one_step_joint_mae_deg"] = float(
                (p[:, 0, v : v + 5] - target[:, 0, v : v + 5]).abs().mean() * 180
            )
            result[f"{name}_one_step_opening_mae"] = float((p[:, 0, v + 5] - target[:, 0, v + 5]).abs().mean())
            result[f"{name}_chunk_state_mse"] = float((p[..., v:] - target[..., v:]).square().mean())
    good = batch["action_valid"].bool().all(1)
    result["successful_action_windows"] = int(good.sum())
    if good.any():
        proposal = model.sample_actions(
            z[good, :h], goals[good], generator=torch.Generator(device=adapter.device).manual_seed(seed)
        )
        result["sampled_action_normalized_mse"] = float(
            ((proposal - actions[good]) / model.action_scale).square().mean()
        )
        result["zero_action_normalized_mse"] = float((actions[good] / model.action_scale).square().mean())
    return result
