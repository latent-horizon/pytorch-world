"""Controlled, simulation-only LeWAM failure investigation and memorization test.

This is a training-task diagnostic, never a held-out benchmark or hardware gate.
Run with ``python -m torch_world.sim2real.investigation --help``.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from torch_world.planning import LeWAMPlanner
from torch_world.sim2real.config import RandomizationConfig
from torch_world.sim2real.control import ClearanceError, Observation
from torch_world.sim2real.data import Episode, LeWAMWindows
from torch_world.sim2real.diagnostics import diagnose_lewam
from torch_world.sim2real.evaluation import restore_run
from torch_world.sim2real.experiment import save_checkpoint, save_video, train
from torch_world.sim2real.models import WorldModelPolicy, make_adapter
from torch_world.sim2real.simulation import SO101Chess


def trace_episode(
    env,
    *,
    seed,
    source,
    target,
    policy_factory=None,
    replay=None,
    capture=None,
    perturbations=None,
    expert_transition="timed",
    include_commanded=False,
):
    """Record a fixed task, including rejected-command evidence before stepping.

    No policy factory/replay means the privileged scripted controller. Geometric
    traces are for diagnosis only; learned policies receive Observation objects.
    Replay actions pass through the same public ``env.step`` as learned actions.
    Optional ``capture(env)`` records presentation views after reset/each step;
    its output is never passed to the policy.
    ``perturbations`` maps control-step indices to replacement actions. These
    exogenous interventions remain dynamics targets but are masked out of action
    imitation. All corrections still pass through the ordinary controller.
    """
    if policy_factory is not None and replay is not None:
        raise ValueError("Choose a policy or replay, not both")
    obs = env.reset(seed=seed, source=source, target=target)
    policy = None if policy_factory is None else policy_factory(env.goal())
    if capture is not None:
        capture(env)
    perturbations = perturbations or {}
    if any(not isinstance(t, int) or t < 0 for t in perturbations):
        raise ValueError("Perturbation steps must be nonnegative integers")
    pixels, states, requested, applied, rows = (
        [(obs.rgb * 255).round().byte()],
        [obs.state(include_commanded)],
        [],
        [],
        [],
    )
    valid = []
    failure, evidence = None, None
    for step in range(env.task.max_steps):
        if replay is not None and step >= len(replay):
            break
        action = (
            replay[step]
            if replay is not None
            else env.expert(transition=expert_transition)
            if policy is None
            else policy(obs)
        )
        proposal = torch.as_tensor(action).tolist()
        action = perturbations.get(step, action)
        row = {
            "step": step,
            "phase": env.phase if policy is None and replay is None else None,
            "tool": env.tool_position.tolist(),
            "piece": env.position.tolist(),
            "opening": env.opening,
            "command_target_gap_mm": float(1000 * np.linalg.norm(env.controller.target - env.tool_position)),
            "command_opening_gap": float(env.controller.opening - env.opening),
            "requested": torch.as_tensor(action).tolist(),
            "proposed": proposal,
            "intervention": step in perturbations,
        }
        rows.append(row)
        before_time = float(env.data.time)
        try:
            obs = env.step(action)
        except ValueError as error:
            failure = str(error)
            if isinstance(error, ClearanceError):
                evidence = {
                    "checked_contacts": error.contacts,
                    "interpolated_opening_contacts": env.controller.clearance_contacts(
                        error.start, error.target, error.opening, start_opening=env.opening
                    ),
                    "measured_start_contacts": env.controller.clearance_contacts(error.start, error.start, env.opening),
                    "physics_advanced_on_rejection": float(env.data.time) != before_time,
                }
            break
        applied.append(torch.tensor(env.last_command.applied, dtype=torch.float32))
        requested.append(torch.as_tensor(action).float())
        valid.append(float(step not in perturbations))
        row["applied"] = applied[-1].tolist()
        pixels.append((obs.rgb * 255).round().byte())
        states.append(obs.state(include_commanded))
        if capture is not None:
            capture(env)
        if policy is not None:
            policy.record_action(env.last_command.applied)
        if env.done:
            break
    metadata = {
        "seed": seed,
        "source": source,
        "target": target,
        "piece": env.piece,
        "color": env.color,
        "randomization": env.episode_parameters,
        "outcome": env.info(),
        "failure": failure,
        "expert_transition": expert_transition if policy is None and replay is None else None,
        "perturbations": {str(k): list(v) for k, v in perturbations.items()},
    }
    episode = Episode(
        torch.stack(pixels),
        torch.stack(states),
        torch.stack(applied) if applied else torch.empty(0, 4),
        torch.stack(requested) if requested else torch.empty(0, 4),
        metadata,
        torch.tensor(valid, dtype=torch.float32),
    )
    report = {
        **metadata,
        "executed_steps": len(applied),
        "attempted_steps": len(rows),
        "rejection": evidence,
        "trace": rows,
        "max_planning_seconds": max(policy.latencies) if policy is not None and policy.latencies else None,
        "mean_planning_seconds": float(np.mean(policy.latencies)) if policy is not None and policy.latencies else None,
    }
    return episode, report


@torch.no_grad()
def action_tracking(adapter, episode, *, seed=17, planner=None):
    """Teacher-forced first-action errors, including active-motion-only errors.

    Supply a planner to compare sampling/integration settings on the same data.
    The default is LeWAM's direct Gaussian-sampled proposal policy.
    """
    if adapter.kind != "lewam":
        raise ValueError("Action tracking requires a LeWAM adapter")
    adapter.eval()
    planner = planner or LeWAMPlanner()
    dataset = LeWAMWindows([episode], adapter.history, adapter.model.chunk_size)
    rng = torch.Generator(device=adapter.device).manual_seed(seed)
    values = []
    for start in range(0, len(dataset), 32):
        samples = [dataset[i] for i in range(start, min(start + 32, len(dataset)))]
        batch = {key: torch.stack([s[key] for s in samples]).to(adapter.device) for key in samples[0]}
        h = adapter.history
        latent = adapter.encode(batch["pixels"][:, :h], batch["states"][:, :h])
        goal = adapter.encode(batch["goal_pixels"][:, None], batch["goal_states"][:, None])[:, 0]
        predicted = planner(adapter.model, latent, goal, generator=rng)[:, 0]
        target = batch["actions"][:, h - 1]
        for p, t in zip(predicted.cpu(), target.cpu()):
            values.append({"predicted": p.tolist(), "target": t.tolist()})
    p = torch.tensor([v["predicted"] for v in values]) / adapter.model.action_scale.cpu()
    t = torch.tensor([v["target"] for v in values]) / adapter.model.action_scale.cpu()
    active = t.abs().amax(-1) > 0.1
    return {
        "windows": len(values),
        "first_action_normalized_mse": float((p - t).square().mean()),
        "zero_first_action_normalized_mse": float(t.square().mean()),
        "active_windows": int(active.sum()),
        "active_first_action_normalized_mse": float((p[active] - t[active]).square().mean()) if active.any() else None,
        "active_zero_first_action_normalized_mse": float(t[active].square().mean()) if active.any() else None,
        "per_coordinate_mse": (p - t).square().mean(0).tolist(),
        "actions": values,
    }


@torch.no_grad()
def observation_aliases(episode, *, history=3, chunk_size=8, action_scale=(1, 1, 1, 0.15), limit=5):
    """Find similar recorded histories whose next demonstrated actions differ.

    Rank within one episode by RGB-history MSE, requiring a normalized action
    difference of at least 0.9 in one coordinate. Report measured-state gaps
    separately; small gaps suggest ambiguity but do not prove unobservability.
    This quadratic diagnostic is intended for short, controlled episodes.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    dataset = LeWAMWindows([episode], history, chunk_size)
    samples = [dataset[i] for i in range(len(dataset))]
    pixels = torch.stack([s["pixels"][:history].flatten() for s in samples])
    states = torch.stack([s["states"][:history] for s in samples])
    actions = torch.stack([s["actions"][history - 1] for s in samples])
    scale = torch.as_tensor(action_scale, dtype=actions.dtype, device=actions.device)
    if scale.shape != (4,) or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Expected four finite positive action scales")
    difference = ((actions[:, None] - actions[None, :]) / scale).abs().amax(-1)
    distance = torch.cdist(pixels, pixels).square() / pixels.size(1)
    valid = torch.ones_like(distance, dtype=torch.bool).triu(1) & (difference >= 0.9)
    distance[~valid] = float("inf")
    pairs = []
    for _ in range(limit):
        if not torch.isfinite(distance).any():
            break
        i, j = divmod(int(distance.argmin()), len(dataset))
        # Recompute selected tiny distances directly, avoiding cancellation in
        # cdist's matrix-multiplication path for nearly identical observations.
        mse = float((pixels[i] - pixels[j]).square().mean())
        gap = (states[i] - states[j]).abs()
        pairs.append(
            {
                "steps": [dataset.index[i][1], dataset.index[j][1]],
                "rgb_history_mse": mse,
                "rgb_history_rmse_255": float(np.sqrt(mse) * 255),
                "max_joint_difference_rad": float(gap[:, :5].max()),
                "max_opening_difference": float(gap[:, 5].max()),
                "actions": [actions[i].tolist(), actions[j].tolist()],
            }
        )
        distance[i, j] = float("inf")
    return {"history": history, "windows": len(dataset), "pairs": pairs}


def investigate(training_run, output, *, source="b2", target="a5", seed=10007, updates=2000, device="cpu"):
    """Audit an unchanged checkpoint, then fit a fresh compact model to one move.

    Three policy RNG seeds test stochastic execution on the *same training task*.
    Reset seeds with randomization disabled are not independent physical scenes.
    The original checkpoint, dataset, controller limits and success rule stay intact.
    """
    if updates < 0:
        raise ValueError("updates must be nonnegative")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    adapter, config, task, calibration = restore_run(training_run, device=device)
    if adapter.kind != "lewam":
        raise ValueError("The investigation requires a LeWAM checkpoint")
    report = {
        "protocol": "fixed-task replay and training-task memorization; not held-out evaluation",
        "checkpoint": str(training_run),
        "checkpoint_sha256": hashlib.sha256((Path(training_run) / "model.pt").read_bytes()).hexdigest(),
        "config": asdict(config),
        "task": asdict(task),
        "randomization": {"enabled": False},
        "source": source,
        "target": target,
        "seed": seed,
        "updates": updates,
        "device": device,
        "physical_results": None,
        "eligible_for_sim2real_gate": False,
    }

    def persist():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    with SO101Chess(config, task, randomization=RandomizationConfig(enabled=False), calibration=calibration) as env:
        settings = {"seed": seed, "source": source, "target": target}
        demo, report["scripted"] = trace_episode(env, **settings)
        torch.save(asdict(demo), output / "demonstration.pt")
        save_video(output / "scripted.gif", demo.pixels, control_dt=config.control_dt)
        persist()
        if not demo.metadata["outcome"]["success"]:
            report["stopped_reason"] = "Scripted reference failed; fix this before attempting memorization"
            persist()
            return report
        report["observation_aliases"] = observation_aliases(
            demo,
            history=adapter.history,
            chunk_size=adapter.model.chunk_size,
            action_scale=adapter.model.action_scale.cpu(),
        )
        for name, actions in (("requested_replay", demo.requested_actions), ("applied_replay", demo.actions)):
            replay, result = trace_episode(env, **settings, replay=actions)
            n = min(len(replay.proprio), len(demo.proprio))
            result["max_state_difference"] = float((replay.proprio[:n] - demo.proprio[:n]).abs().max())
            result["same_length"] = len(replay.actions) == len(demo.actions)
            report[name] = result
            persist()
        if any(
            not report[name]["outcome"]["success"]
            or not report[name]["same_length"]
            or report[name]["max_state_difference"] > 1e-6
            for name in ("requested_replay", "applied_replay")
        ):
            report["stopped_reason"] = "Action replay diverged; resolve the controller/data contract before training"
            persist()
            return report
        report["original_tracking"] = action_tracking(adapter, demo)
        original, report["original_policy"] = trace_episode(
            env, **settings, policy_factory=lambda goal: WorldModelPolicy(adapter, goal, seed=seed)
        )
        save_video(output / "original-policy.gif", original.pixels, control_dt=config.control_dt)
        persist()
        if not updates:
            return report
        torch.manual_seed(17)
        fresh = make_adapter(
            "lewam",
            size=config.size,
            smoke=True,
            proprioception=True,
            chunk_size=8,
            action_scale=(1, 1, 1, config.max_gripper_step),
        ).to(device)
        metrics = []

        def progress(event):
            metrics.append(event)
            print(json.dumps(event), flush=True)

        report["memorization"] = {"training_episodes": 1, "model": fresh.recipe, "seed": 17}
        report["memorization"]["losses"] = train(
            fresh, [demo], updates=updates, batch_size=32, seed=17, progress=progress
        )
        fresh.eval()
        save_checkpoint(output / "memorized.pt", fresh, config, task, calibration)
        report["memorization"]["training_metrics"] = metrics
        report["memorization"]["diagnostics"] = diagnose_lewam(fresh, [demo], windows_per_episode=32)
        report["memorization"]["tracking"] = action_tracking(fresh, demo)
        report["memorization"]["trials"] = []
        persist()
        # Compare the training terminal observation with the rendered deployment
        # goal: no intermediate targets or expert actions enter either policy.
        terminal = Observation(demo.pixels[-1].float() / 255, demo.proprio[-1, :5], float(demo.proprio[-1, 5]), 0)
        for goal_mode in ("rendered", "demonstration_terminal"):
            for policy_seed in (17, 18, 19):

                def factory(goal, goal_mode=goal_mode, policy_seed=policy_seed):
                    return WorldModelPolicy(
                        fresh, terminal if goal_mode == "demonstration_terminal" else goal, seed=policy_seed
                    )

                rollout, result = trace_episode(env, **settings, policy_factory=factory)
                result.update(goal_mode=goal_mode, policy_seed=policy_seed)
                report["memorization"]["trials"].append(result)
                if policy_seed == 17:
                    save_video(output / f"memorized-{goal_mode}.gif", rollout.pixels, control_dt=config.control_dt)
                print(goal_mode, policy_seed, result["outcome"], result["failure"], flush=True)
                persist()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_run", type=Path)
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory; existing results are never overwritten"
    )
    parser.add_argument("--source", default="b2")
    parser.add_argument("--target", default="a5")
    parser.add_argument("--seed", type=int, default=10007)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    investigate(
        args.training_run,
        args.output,
        source=args.source,
        target=args.target,
        seed=args.seed,
        updates=args.updates,
        device=args.device,
    )


if __name__ == "__main__":
    main()
