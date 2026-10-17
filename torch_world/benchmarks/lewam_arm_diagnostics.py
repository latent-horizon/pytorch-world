"""Replay and closed-loop diagnostics for the owned-arm LeWAM experiment.

Simulator states and expert commands are used only for auditing, never as inputs
or replacement actions for the visual policy.
"""

import torch

from torch_world.benchmarks.lewam_arm import demonstration_task, environment


def audit_demonstrations(data, config, *, seeds=None):
    """Replay controls and test that rendering goals does not mutate live physics.

    Check a deterministic prefix of the training split by default. Errors are
    reported rather than hidden by relaxed tolerances or discarded episodes.
    """
    seeds = list(range(config.seed, config.seed + min(3, len(data["pixels"])))) if seeds is None else list(seeds)
    rows = []
    with environment(config) as env:
        for seed in seeds:
            index = seed - config.seed
            if not 0 <= index < len(data["pixels"]):
                raise ValueError("Replay seeds must belong to the training split")
            start, goal = demonstration_task(seed, config)
            env.goal = goal[None]
            obs = env.reset(start[None])
            state_error = float((env.state[0] - data["states"][index, 0]).abs().max())
            pixel_error = float(((obs[0] * 255).round() - data["pixels"][index, 0]).abs().max())
            before = env.state.clone()
            env.render(env.goal)
            live = torch.tensor([*env.data[0].qpos, *env.data[0].qvel], dtype=torch.float32)
            render_error = float((live - before[0]).abs().max())
            for step, action in enumerate(data["actions"][index]):
                obs = env.step(action[None])
                state_error = max(state_error, float((env.state[0] - data["states"][index, step + 1]).abs().max()))
                pixel_error = max(
                    pixel_error,
                    float(((obs[0] * 255).round() - data["pixels"][index, step + 1]).abs().max()),
                )
            rows.append(
                {
                    "seed": seed,
                    "max_state_error": state_error,
                    "max_pixel_error_uint8": pixel_error,
                    "goal_render_state_error": render_error,
                    "final_success": bool(env.success(env.state, env.goal).item()),
                }
            )
        physics = {
            "timestep": float(env.model.opt.timestep),
            "control_period": float(env.model.opt.timestep * env.substeps),
            "pose_tolerance": list(env.scene.tolerance),
            "velocity_tolerance": env.scene.velocity_tolerance,
        }
    phases = []
    for step in range(min(6, config.steps - 1)):
        delta = (data["pixels"][:, step + 1].float() - data["pixels"][:, step].float()) / 255
        phases.append(
            {
                "step": step,
                "pixel_change_rms": float(delta.square().mean().sqrt()),
                "expert_action_mean": data["actions"][:, step].mean(0).tolist(),
                "expert_action_std": data["actions"][:, step].std(0, unbiased=False).tolist(),
            }
        )
    return {"replay": rows, "physics": physics, "launch_phases": phases}


def plot_rollout(trace, *, trial=0):
    """Plot one preselected rollout, including expert labels on visited states."""
    import matplotlib.pyplot as plt

    states = torch.tensor(trace["states"])[:, trial]
    actions = torch.tensor(trace["actions"])[:, trial]
    expert = torch.tensor(trace["expert_actions"])[:, trial]
    goal = trace["goals"][trial]
    fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharex=True)
    for joint, name in enumerate(("Shoulder", "Elbow")):
        axes[0, joint].plot(states[:, joint], label="Policy pose")
        axes[0, joint].axhline(goal[joint], color="black", linestyle="--", label="Goal")
        axes[0, joint].set(title=name, ylabel="Angle (rad)")
        axes[1, joint].plot(actions[:, joint], label="Policy action")
        axes[1, joint].plot(expert[:, joint], linestyle="--", label="Expert at visited state")
        axes[1, joint].set(xlabel="Control step", ylabel="Normalized torque")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle(f"First recorded task · seed {trace['seeds'][trial]}")
    fig.tight_layout()
    return fig


def result_table(reports, *, controllers=None):
    """Compact comparable outcomes; each report retains its own protocol/budget."""
    controllers = controllers or ("lewam_single", "lewam_mean", "shuffled_goal_mean", "state_policy", "expert")
    rows = ["| Experiment | Controller | Reached | Settled |", "|---|---|---:|---:|"]
    for label, report in reports.items():
        for name in controllers:
            if name not in report["evaluation"]:
                continue
            trials = report["evaluation"][name]
            reached = sum(t["success"] for t in trials)
            settled = sum(t["final_success"] for t in trials)
            rows.append(f"| {label} | {name} | {reached}/{len(trials)} | {settled}/{len(trials)} |")
    return "\n".join(rows)
