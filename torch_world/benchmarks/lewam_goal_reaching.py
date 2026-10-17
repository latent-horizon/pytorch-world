"""Sampled-future goal training and gradient planning in our owned robotic arm.

Protocol adapted from MinghaoFu/lewam f57659f; not a DMControl reproduction.
Evaluation restores a held-out trajectory state, its future goal image, and the
same visible collection target in both views. Native settling and pose-only
success are reported separately; neither replaces the environment predicate.
"""

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from torch_world.benchmarks.lewam_arm import ArmConfig, collect, environment, expert, fit, normalize_pixels
from torch_world.benchmarks.lewam_arm_diagnostics import audit_demonstrations
from torch_world.planning import LeWAMHorizonPlanner


@torch.no_grad()
def sample_subgoals(data, seeds, *, offset=25, start_indices=None):
    """Deterministic early-motion starts; evaluation trajectories are never trained on."""
    if offset < 1 or offset >= data["states"].shape[1]:
        raise ValueError("Invalid future goal offset")
    starts = torch.tensor(
        [np.random.default_rng(seed).integers(0, min(21, data["states"].shape[1] - offset)) for seed in seeds]
    )
    if start_indices is not None:
        starts = torch.as_tensor(start_indices, dtype=torch.long)
        if starts.shape != (len(seeds),) or (starts < 0).any() or (starts + offset >= data["states"].shape[1]).any():
            raise ValueError("Start indices must keep every goal inside its episode")
    rows = torch.arange(len(seeds))
    return starts, data["states"][rows, starts], data["states"][rows, starts + offset]


def evaluate_goals(
    model,
    config,
    data,
    seeds,
    output,
    device,
    *,
    offset=25,
    budget=50,
    proposals=32,
    refinements=5,
    start_indices=None,
    controllers=None,
    rng_seed=999,
):
    """Compare policy and learned planning on exactly the same unseen subgoals."""
    from PIL import Image

    if config.goal_mode != "sampled" or config.history_stride != config.frameskip:
        raise ValueError("Goal-reaching evaluation requires sampled goals and strided history")
    if offset > config.goal_horizon * config.frameskip or not config.frameskip <= budget <= config.steps:
        raise ValueError("Evaluation horizon must fit training horizons; budget must cover a control block")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start_indices, starts, goals = sample_subgoals(data, seeds, offset=offset, start_indices=start_indices)
    result = {}
    allowed = ("policy", "planner", "shuffled_goal_planner", "expert", "zero")
    controllers = allowed if controllers is None else tuple(controllers)
    if not controllers or any(name not in allowed for name in controllers):
        raise ValueError("Unknown or empty controller selection")
    for name in controllers:
        visual = name not in ("expert", "zero")
        planner = LeWAMHorizonPlanner(num_samples=proposals, horizon=config.goal_horizon, refinement_steps=refinements)
        with environment(config, batch_size=len(seeds), observation="rgb" if visual else "state") as env:
            env.goal = goals.clone()
            env.set_visual_goal(data["goal"])
            obs = env.reset(starts)
            pose_initial = env.distance(env.state, env.goal) < 1
            pose_hit = pose_initial.clone()
            native_initial = env.success(env.state, env.goal)
            native_hit = native_initial.clone()
            held = torch.ones_like(native_hit)
            trace = {
                "seeds": list(seeds),
                "goals": goals.tolist(),
                "visual_goals": data["goal"].tolist(),
                "states": [env.state.tolist()],
                "actions": [],
                "expert_actions": [],
            }
            rng = torch.Generator(device=device).manual_seed(rng_seed)
            frames, pictures, planned_costs = [], [], []
            if visual:
                frames.append(normalize_pixels(obs.to(device), config))
                goal_image = env.render(goals)
                # Rendering/restoration must reproduce the recorded goal view, including the target.
                saved = data["pixels"][torch.arange(len(seeds)), start_indices + offset].float() / 255
                if not torch.allclose(goal_image, saved, atol=1 / 255):
                    raise RuntimeError("Rendered subgoal does not match its held-out recorded image")
                with torch.no_grad():
                    target = model.encode(normalize_pixels(goal_image.to(device), config)[:, None])[:, 0]
                if name == "shuffled_goal_planner":
                    target = target.roll(1, 0)
            for step in range(budget):
                if visual:
                    if step % config.frameskip == 0:
                        times = [step + i * config.frameskip for i in range(1 - config.history, 1)]
                        history = torch.stack([frames[max(0, t)] for t in times], 1)
                        pad = torch.tensor([t < 0 for t in times], device=device)[None].expand(len(seeds), -1)
                        left = history.new_full((len(seeds),), max(1.0, (offset - step) / config.frameskip))
                        with torch.no_grad():
                            z = model.encode(history)
                        if name == "policy":
                            with torch.no_grad():
                                cached = model.sample_actions(
                                    z,
                                    target,
                                    history_pad=pad,
                                    remaining=left.clamp_max(config.goal_horizon) / config.goal_horizon,
                                    generator=rng,
                                )
                        else:
                            cached = planner(model, z, target, history_pad=pad, steps_to_goal=left, generator=rng)
                            planned_costs.append(
                                {key: value.cpu().tolist() for key, value in planner.last_diagnostics.items()}
                            )
                        cached = cached.cpu()
                    action = cached[:, step % config.frameskip]
                else:
                    action = expert(env, gain=config.expert_gain) if name == "expert" else torch.zeros(len(seeds), 2)
                trace["actions"].append(action.tolist())
                trace["expert_actions"].append(expert(env, gain=config.expert_gain).tolist())
                obs = env.step(action)
                trace["states"].append(env.state.tolist())
                native_now = env.success(env.state, env.goal)
                native_hit |= native_now
                if step >= budget - min(10, budget):
                    held &= native_now
                pose_hit |= env.distance(env.state, env.goal) < 1
                if visual:
                    frames.append(normalize_pixels(obs.to(device), config))
                    if name == "planner":
                        pair = torch.cat((obs[0], goal_image[0]), -1).permute(1, 2, 0)
                        pictures.append(
                            Image.fromarray((pair.numpy() * 255).round().astype(np.uint8)).resize((512, 256))
                        )
            final = env.success(env.state, env.goal)
            distances = env.distance(env.state, env.goal)
            result[name] = [
                {
                    "seed": seed,
                    "start_step": int(start_indices[i]),
                    "pose_initial_success": bool(pose_initial[i]),
                    "pose_success": bool(pose_hit[i]),
                    "initial_success": bool(native_initial[i]),
                    "success": bool(native_hit[i]),
                    "final_success": bool(final[i]),
                    "hold_success": bool(held[i]),
                    "final_pose_success": bool(distances[i] < 1),
                    "final_normalized_distance": float(distances[i]),
                }
                for i, seed in enumerate(seeds)
            ]
            (output / f"trace-{name}.json").write_text(json.dumps({**trace, "planning_costs": planned_costs}))
            if pictures:
                pictures[0].save(output / "rollout.gif", save_all=True, append_images=pictures[1:], duration=50, loop=0)
        print(
            f"{name}: pose {int(pose_hit.sum())}/{len(seeds)}, native reached {int(native_hit.sum())}/{len(seeds)}, "
            f"native final {int(final.sum())}/{len(seeds)}",
            flush=True,
        )
    return result


def run_goal_reaching(
    output,
    config,
    *,
    device=None,
    initial_checkpoint=None,
    data_path=None,
    proposals=32,
    refinements=5,
    offset=25,
    budget=50,
):
    """Train sampled goals, then evaluate the direct policy and multi-step planner."""
    if config.goal_mode != "sampled":
        raise ValueError("Use ArmConfig(goal_mode='sampled', history_stride=frameskip)")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2))
    source_root = Path(__file__).resolve().parents[1]
    sources = {}
    for path in (
        "benchmarks/lewam_goal_reaching.py",
        "benchmarks/lewam_arm.py",
        "benchmarks/lewam_arm_diagnostics.py",
        "models/jepa/_lewam.py",
        "models/jepa/_lewam_encoder.py",
        "planning/lewam_horizon.py",
        "envs/physics_tasks.py",
        "envs/physics_scenes.py",
    ):
        raw = (source_root / path).read_bytes()
        sources[path] = hashlib.sha256(raw).hexdigest()
        destination = output / "source" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    seeds = {
        key: list(range(config.seed + shift, config.seed + shift + count))
        for key, shift, count in (
            ("train", 0, config.train_episodes),
            ("validation", 10000, config.validation_episodes),
            ("test", config.test_seed_offset, config.test_episodes),
        )
    }
    if len(set().union(*map(set, seeds.values()))) != sum(map(len, seeds.values())):
        raise ValueError("Episode seeds overlap")
    if data_path is None:
        data = {key: collect(config, seeds[key]) for key in ("train", "validation")}
        torch.save(data, output / "episodes.pt")
    else:
        data_path = Path(data_path)
        saved = ArmConfig(**json.loads((data_path.parent / "config.json").read_text()))
        for key in (
            "seed",
            "size",
            "steps",
            "train_episodes",
            "validation_episodes",
            "expert_gain",
            "recovery_fraction",
            "goal_marker",
            "diverse_motion",
        ):
            if getattr(saved, key) != getattr(config, key):
                raise ValueError(f"Cached demonstrations disagree on {key}")
        data = torch.load(data_path, weights_only=True)
        (output / "data-source.json").write_text(json.dumps({"path": str(data_path.resolve())}))
    audit = audit_demonstrations(data["train"], config)
    if any(
        r["max_state_error"] > 1e-5
        or r["max_pixel_error_uint8"] > 1
        or r["goal_render_state_error"] > 1e-7
        or not r["final_success"]
        for r in audit["replay"]
    ):
        raise RuntimeError("Replay audit failed")
    model, _, curve, selected = fit(
        data["train"], data["validation"], config, output, device, initial_checkpoint=initial_checkpoint
    )
    # Held-out trajectories are generated only after fitting/selection; no test windows are trained on.
    test = collect(config, seeds["test"])
    evaluation = evaluate_goals(
        model,
        config,
        test,
        seeds["test"],
        output,
        device,
        offset=offset,
        budget=budget,
        proposals=proposals,
        refinements=refinements,
    )
    import mujoco

    result = {
        "protocol": "lewam-owned-arm-sampled-goal-planning-v1",
        "environment": "ObstacleArm-v0",
        "reference_commit": "f57659fb26858082975794d37a8f3d380ddc1fa7",
        "config": asdict(config),
        "planning": {"proposals": proposals, "refinements": refinements, "goal_offset": offset, "budget": budget},
        "curve": curve,
        "selected_updates": selected,
        "episode_seeds": seeds,
        "evaluation": evaluation,
        "environment_audit": audit,
        "source_sha256": sources,
        "host": platform.node(),
        "device": device,
        "versions": {"torch": torch.__version__, "mujoco": mujoco.__version__},
        "initial_checkpoint_sha256": None
        if initial_checkpoint is None
        else hashlib.sha256(Path(initial_checkpoint).read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256((output / "model.pt").read_bytes()).hexdigest(),
        "seconds": time.perf_counter() - started,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2))
    return result


def goal_reaching_config(**overrides):
    """Reduced-budget, source-aligned protocol for the owned arm example."""
    return replace(
        ArmConfig(
            reference=True,
            size=224,
            history=2,
            history_stride=5,
            frameskip=5,
            chunk_size=10,
            flow_steps=8,
            execution_interval=5,
            expert_gain=2,
            goal_marker=True,
            goal_mode="sampled",
            goal_horizon=10,
            updates=1000,
            batch_size=16,
            phase_validation=True,
            test_episodes=8,
            test_seed_offset=40000,
        ),
        **overrides,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--test-episodes", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    run_goal_reaching(
        args.output,
        goal_reaching_config(updates=args.updates, test_episodes=args.test_episodes),
        device=args.device,
        initial_checkpoint=args.initial_checkpoint,
        data_path=args.data,
    )


if __name__ == "__main__":
    main()
