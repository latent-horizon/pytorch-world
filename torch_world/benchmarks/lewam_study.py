"""Evaluate a completed matched LeWAM study, retaining every trial and trace."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from torch_world.benchmarks.lewam_arm import ArmConfig, collect, environment, make_model
from torch_world.benchmarks.lewam_arm_diagnostics import audit_demonstrations
from torch_world.benchmarks.lewam_goal_reaching import evaluate_goals, goal_reaching_config
from torch_world.benchmarks.lewam_matched import save_json
from torch_world.benchmarks.lewam_prediction_audit import audit_predictions
from torch_world.benchmarks.lewam_reference import ReferenceLeWAM


def summarize_trials(trials):
    """Include initial successes and conditional results, never silently drop them."""
    result = {"count": len(trials)}
    for key in ("pose_initial_success", "initial_success", "pose_success", "success", "final_success", "hold_success"):
        result[key] = sum(t[key] for t in trials)
    eligible = [t for t in trials if not t["pose_initial_success"]]
    result["initially_pose_unsatisfied"] = {
        "count": len(eligible),
        "pose_success": sum(t["pose_success"] for t in eligible),
        "native_success": sum(t["success"] for t in eligible),
    }
    return result


def evaluate_study(training, reference, output, *, device="cuda", cases=50, batch_size=10):
    torch.set_num_threads(4)
    training, output = Path(training), Path(output)
    if cases < 1 or batch_size < 1:
        raise ValueError("Positive evaluation budgets required")
    trained = json.loads((training / "training.json").read_text())
    if not trained.get("complete"):
        raise ValueError("Training has not completed")
    output.mkdir(parents=True, exist_ok=False)
    config = ArmConfig(**trained["config"])
    data = torch.load(trained["data_path"], weights_only=True)
    flat = data["train"]["actions"].flatten(0, 1)
    mean, std = flat.mean(0), flat.std(0).clamp_min(1e-6)
    models = {
        "local": make_model(config, action_mean=mean, action_std=std),
        "upstream": ReferenceLeWAM(config, reference, mean, std),
    }
    for name, model in models.items():
        actual = hashlib.sha256((training / f"{name}.pt").read_bytes()).hexdigest()
        if actual != trained["checkpoint_sha256"][name]:
            raise ValueError("Checkpoint hash mismatch")
        model.load_state_dict(torch.load(training / f"{name}.pt", weights_only=True, map_location="cpu")["model"])
        model.to(device).eval()
    report = {
        "protocol": "lewam-owned-arm-matched-evaluation-v1",
        "training": trained,
        "evaluation_seeds": [42, 0, 1],
        "cases_per_seed": cases,
        "groups": {},
        "summary": {},
        "complete": False,
    }
    report["prediction_audit"] = audit_predictions(models, config, data["validation"], output, device=device)
    for split in ("training_trajectories", "unseen_trajectories"):
        report["groups"][split] = {}
        for index, seed in enumerate(report["evaluation_seeds"]):
            rng = np.random.default_rng(seed)
            if split == "training_trajectories":
                ids = torch.tensor(rng.choice(config.train_episodes, cases, replace=cases > config.train_episodes))
                episodes = {k: v[ids] for k, v in data["train"].items()}
                seeds = (ids + config.seed).tolist()
            else:
                seeds = list(range(config.seed + 50000 + index * 1000, config.seed + 50000 + index * 1000 + cases))
                episodes = collect(config, seeds)
            starts = torch.tensor(rng.integers(0, config.steps + 1 - 25, size=cases))
            group = {name: {} for name in models}
            for begin in range(0, cases, batch_size):
                end = min(cases, begin + batch_size)
                batch = {k: v[begin:end] for k, v in episodes.items()}
                for name, model in models.items():
                    destination = output / split / str(seed) / name / str(begin)
                    controllers = None if name == "local" else ("policy", "planner", "shuffled_goal_planner")
                    trials = evaluate_goals(
                        model,
                        config,
                        batch,
                        seeds[begin:end],
                        destination,
                        device,
                        start_indices=starts[begin:end],
                        controllers=controllers,
                        rng_seed=seed * 1000 + begin,
                    )
                    for controller, rows in trials.items():
                        group[name].setdefault(controller, []).extend(rows)
            for baseline in ("expert", "zero"):
                group["upstream"][baseline] = group["local"][baseline]
            report["groups"][split][str(seed)] = group
            save_json(output / "results.json", report)
    for split, seeds in report["groups"].items():
        report["summary"][split] = {}
        for name in models:
            report["summary"][split][name] = {
                controller: summarize_trials([t for g in seeds.values() for t in g[name][controller]])
                for controller in ("policy", "planner", "shuffled_goal_planner", "expert", "zero")
            }
    report["complete"] = True
    save_json(output / "results.json", report)
    return report


def prepare_data(output, *, diverse_motion=False):
    """Collect a separate coverage dataset; leave the matched baseline data intact."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = goal_reaching_config(diverse_motion=diverse_motion)
    save_json(output / "config.json", asdict(config))
    data = {
        split: collect(config, list(range(config.seed + offset, config.seed + offset + count)))
        for split, offset, count in (
            ("train", 0, config.train_episodes),
            ("validation", 10000, config.validation_episodes),
        )
    }
    audit = audit_demonstrations(data["train"], config)
    if any(r["max_state_error"] > 1e-5 or r["max_pixel_error_uint8"] > 1 for r in audit["replay"]):
        raise RuntimeError("Collected trajectories failed replay")
    save_json(output / "replay.json", audit)
    torch.save(data, output / "episodes.pt")
    with environment(config, observation="state") as env:
        held = {
            split: env.success(
                value["states"][:, -10:].reshape(-1, 4), value["goal"][:, None].expand(-1, 10, -1).reshape(-1, 4)
            )
            .reshape(len(value["goal"]), 10)
            .all(1)
            .tolist()
            for split, value in data.items()
        }
    save_json(
        output / "coverage.json",
        {
            "modes": ["approach", "reverse", "recovery", "brake_and_hold"] if diverse_motion else ["approach"],
            "ten_frame_hold": held,
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training", type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prepare-diverse", action="store_true")
    p.add_argument("--prepare-data", action="store_true")
    p.add_argument("--cases", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if a.prepare_diverse or a.prepare_data:
        prepare_data(a.output, diverse_motion=a.prepare_diverse)
    else:
        if a.training is None or a.reference is None:
            p.error("--training and --reference are required for evaluation")
        evaluate_study(a.training, a.reference, a.output, device=a.device, cases=a.cases, batch_size=a.batch_size)


if __name__ == "__main__":
    main()
