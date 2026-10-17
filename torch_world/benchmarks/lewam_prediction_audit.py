"""Compare learned action rankings with real, resettable MuJoCo outcomes.

Candidate controls are identical across models, generated from held-out recorded
controls plus perturbations. No privileged quantities enter visual predictions.
"""

import json
from pathlib import Path

import torch

from torch_world.benchmarks.lewam_arm import environment, normalize_pixels


def ranking_metrics(predicted, actual):
    """Per-task pairwise agreement and regret; ties are excluded from agreement."""
    if predicted.shape != actual.shape or predicted.ndim != 2 or predicted.shape[1] < 2:
        raise ValueError("Expected matching (tasks,candidates>=2) costs")
    if not torch.isfinite(predicted).all() or not torch.isfinite(actual).all():
        raise ValueError("Costs must be finite")
    n = predicted.shape[1]
    a, b = torch.triu_indices(n, n, 1)
    dp = predicted[:, a] - predicted[:, b]
    da = actual[:, a] - actual[:, b]
    valid = (dp.abs() > 1e-8) & (da.abs() > 1e-8)
    selected = actual.gather(1, predicted.argmin(1)[:, None])[:, 0]
    count = valid.sum(1)
    agreement = ((dp * da > 0) & valid).sum(1) / count.clamp_min(1)
    return {
        "pairwise_agreement": [float(x) if int(k) else None for x, k in zip(agreement, count)],
        "selected_actual_distance": selected.tolist(),
        "mean_candidate_distance": actual.mean(1).tolist(),
        "oracle_candidate_distance": actual.min(1).values.tolist(),
        "selection_regret": (selected - actual.min(1).values).tolist(),
        "valid_pairs": count.tolist(),
    }


@torch.no_grad()
def audit_predictions(models, config, data, output, *, device="cuda", tasks=8, candidates=8):
    """Open-loop prediction/ranking audit at 5, 10 and 25 physical controls."""
    if config.frameskip != 5 or config.history != 2 or candidates < 4:
        raise ValueError("Audit requires two frames, frameskip five and >=4 candidates")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = torch.arange(min(tasks, len(data["states"])))
    count = len(rows)
    times = torch.arange(count) % 4 * 5
    states = data["states"][rows, times].repeat_interleave(candidates, 0)
    goals = data["states"][rows, times + 25].repeat_interleave(candidates, 0)
    visible = data["goal"][rows].repeat_interleave(candidates, 0)
    rng = torch.Generator().manual_seed(8042)
    controls = data["actions"][rows[:, None], times[:, None] + torch.arange(25)]
    controls = controls[:, None].expand(-1, candidates, -1, -1).clone()
    scale = torch.linspace(0, 0.25, candidates)[None, :, None, None]
    controls += torch.randn(controls.shape, generator=rng) * scale
    controls[:, -2] = 0
    controls[:, -1] = -controls[:, 0]
    controls = controls.clamp(-1, 1).reshape(count * candidates, 25, 2)
    history_times = torch.stack(((times - 5).clamp_min(0), times), 1)
    history = data["pixels"][rows[:, None], history_times].float() / 255
    history = normalize_pixels(history, config).repeat_interleave(candidates, 0).to(device)
    pad = (
        torch.stack((times < 5, torch.zeros_like(times, dtype=torch.bool)), 1)
        .repeat_interleave(candidates, 0)
        .to(device)
    )
    goals_image = data["pixels"][rows, times + 25].float() / 255
    goals_image = normalize_pixels(goals_image, config).to(device)
    state_histories, targets = {}, {}
    for name, model in models.items():
        model.eval()
        state_histories[name] = model.encode(history)
        targets[name] = model.encode(goals_image[:, None])[:, 0].repeat_interleave(candidates, 0)
    report = {
        "tasks": count,
        "candidates": candidates,
        "start_steps": times.tolist(),
        "source": "first validation episodes; not used for checkpoint selection",
        "horizons": {},
    }
    with environment(config, batch_size=count * candidates) as env:
        env.goal = goals
        env.set_visual_goal(visible)
        env.reset(states)
        for step in range(25):
            obs = env.step(controls[:, step])
            if (step + 1) % 5:
                continue
            block = controls[:, step - 4 : step + 1].to(device)
            for name, model in models.items():
                padded = torch.cat((block, block.new_zeros(len(block), model.chunk_size - 5, 2)), 1)
                z = model.predict_future(state_histories[name], padded, history_pad=pad)[:, :1]
                state_histories[name] = torch.cat((state_histories[name][:, 1:], z), 1)
            pad = torch.cat((pad[:, 1:], torch.zeros_like(pad[:, :1])), 1)
            if step + 1 not in (5, 10, 25):
                continue
            actual = env.distance(env.state, env.goal).reshape(count, candidates)
            metrics = {}
            for name, model in models.items():
                predicted = state_histories[name][:, -1]
                true = model.encode(normalize_pixels(obs.to(device), config)[:, None])[:, 0]
                latent_cost = (predicted - targets[name]).square().mean(-1).reshape(count, candidates).cpu()
                real_latent_cost = (true - targets[name]).square().mean(-1).reshape(count, candidates).cpu()
                mse = (predicted - true).square().mean()
                variance = true.var(0, unbiased=False).mean()
                metrics[name] = {
                    "latent_mse": float(mse),
                    "actual_latent_variance": float(variance),
                    "variance_normalized_mse": float(mse / variance.clamp_min(1e-8)),
                    "imagined_cost": latent_cost.tolist(),
                    "actual_pose_distance": actual.tolist(),
                    "predicted_ranking": ranking_metrics(latent_cost, actual),
                    "encoded_actual_ranking": ranking_metrics(real_latent_cost, actual),
                }
            report["horizons"][str(step + 1)] = metrics
    (output / "prediction-audit.json").write_text(json.dumps(report, indent=2))
    return report
