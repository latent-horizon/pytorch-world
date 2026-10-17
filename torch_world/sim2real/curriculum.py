"""Matched LeWAM memory/recovery ablations on the SO101 pawn task.

Separate preparation, training and evaluation processes keep physics independent
of GPU training and preserve every completed trial. This is a controlled first
curriculum stage, not a sim2real qualification benchmark.
"""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from torch_world.planning import LeWAMPlanner
from torch_world.sim2real.config import RandomizationConfig
from torch_world.sim2real.data import Episode, LeWAMWindows
from torch_world.sim2real.evaluation import restore_run
from torch_world.sim2real.experiment import save_checkpoint, save_video, train
from torch_world.sim2real.investigation import action_tracking, trace_episode
from torch_world.sim2real.models import WorldModelPolicy, make_adapter
from torch_world.sim2real.reference import reference_setup
from torch_world.sim2real.simulation import SO101Chess

VARIANTS = {
    "h3_clean": {"history": 3, "dataset": "clean", "command_state": False},
    "h16_clean": {"history": 16, "dataset": "clean", "command_state": False},
    "h16_recovery": {"history": 16, "dataset": "recovery", "command_state": False},
    "h16_recovery_command": {"history": 16, "dataset": "recovery", "command_state": True},
    "h3_observed": {"history": 3, "dataset": "observed", "command_state": False},
}
TASK = {"seed": 10007, "source": "b2", "target": "a5"}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def recovery_schedule(seed, step):
    """One bounded lateral/upward command and small opening disturbance."""
    rng = np.random.default_rng(seed)
    angle, magnitude = rng.uniform(0, 2 * np.pi), rng.uniform(0.2, 0.6)
    return {
        step: [
            float(magnitude * np.cos(angle)),
            float(magnitude * np.sin(angle)),
            float(rng.uniform(0, 0.3)),
            float(rng.uniform(-0.04, 0.04)),
        ]
    }


def evaluation_cases():
    """Frozen before training; unseen disturbance draws on the same board move."""
    return [{"name": "nominal", "seed": 50000, "perturbations": {}}] + [
        {"name": f"disturbance_{step}", "seed": 50001 + i, "perturbations": recovery_schedule(50001 + i, step)}
        for i, step in enumerate((4, 16, 50, 73))
    ]


def prepare(output, *, size=64, recovery_count=12):
    """Collect all attempted demonstrations and independent scripted controls."""
    if recovery_count < 1:
        raise ValueError("recovery_count must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config, task = reference_setup(size=size)
    report = {
        "protocol": "matched memory, recovery, commanded-state and observed-transition ablations",
        "config": asdict(config),
        "task": asdict(task),
        "training_task": TASK,
        "variants": VARIANTS,
        "evaluation_cases": evaluation_cases(),
        "model_seeds": [17, 29, 43],
        "policy_seeds": [101, 202, 303],
        "updates": 2000,
        "batch_size": 32,
        "noise_scales": [1.0, 0.0],
        "physical_results": None,
        "eligible_for_sim2real_gate": False,
        "scope": "one white pawn move; new disturbance draws are not new tasks or domain randomization",
        "training": {},
        "controls": [],
    }
    episodes = {}
    with SO101Chess(config, task, randomization=RandomizationConfig(enabled=False)) as env:
        for dataset, transition in (("clean", "timed"), ("observed", "observed")):
            episode, result = trace_episode(env, **TASK, expert_transition=transition, include_commanded=True)
            episodes[dataset] = [episode]
            report["training"][dataset] = [result]
            save_video(output / f"{dataset}-scripted.gif", episode.pixels, control_dt=config.control_dt)
            print(dataset, result["outcome"], flush=True)
        if not episodes["clean"][0].metadata["outcome"]["success"]:
            raise RuntimeError("The fixed scripted baseline must succeed before training")
        episodes["recovery"] = [episodes["clean"][0]]
        report["training"]["recovery"] = []
        for i in range(recovery_count):
            perturbations = recovery_schedule(30000 + i, (3, 14, 32, 49, 63, 74)[i % 6])
            episode, result = trace_episode(
                env, **{**TASK, "seed": 30000 + i}, perturbations=perturbations, include_commanded=True
            )
            episodes["recovery"].append(episode)
            report["training"]["recovery"].append(result)
            print("recovery", i, result["outcome"], result["failure"], flush=True)
        for transition in ("timed", "observed"):
            for case in evaluation_cases():
                _, result = trace_episode(
                    env,
                    **{**TASK, "seed": case["seed"]},
                    perturbations=case["perturbations"],
                    expert_transition=transition,
                    include_commanded=True,
                )
                result.update(case=case["name"], transition=transition)
                report["controls"].append(result)
                print("control", transition, case["name"], result["outcome"], flush=True)
    torch.save({k: [asdict(e) for e in group] for k, group in episodes.items()}, output / "episodes.pt")
    report["dataset_sha256"] = hashlib.sha256((output / "episodes.pt").read_bytes()).hexdigest()
    write_json(output / "protocol.json", report)
    return report


def matched_adapter(variant, seed, size):
    """Share equal-shaped initial tensors and the most recent history positions."""
    setting = VARIANTS[variant]
    torch.manual_seed(seed)
    reference = make_adapter("lewam", size=size, history=3, smoke=True, proprioception=True, chunk_size=8)
    if variant in ("h3_clean", "h3_observed"):
        return reference
    torch.manual_seed(seed)
    adapter = make_adapter(
        "lewam",
        size=size,
        history=setting["history"],
        smoke=True,
        proprioception=True,
        command_state=setting["command_state"],
        chunk_size=8,
    )
    common = reference.state_dict()
    own = adapter.state_dict()
    for key, value in common.items():
        if own[key].shape == value.shape:
            own[key].copy_(value)
    own["model.history_position"][:, -3:].copy_(common["model.history_position"])
    adapter.load_state_dict(own)
    return adapter


def train_variant(prepared, output, variant, seed, *, device="cpu", updates=None):
    prepared, output = Path(prepared), Path(output)
    protocol = json.loads((prepared / "protocol.json").read_text())
    if variant not in VARIANTS or seed not in protocol["model_seeds"]:
        raise ValueError("Variant/seed must belong to the predeclared protocol")
    if hashlib.sha256((prepared / "episodes.pt").read_bytes()).hexdigest() != protocol["dataset_sha256"]:
        raise ValueError("Prepared dataset checksum differs")
    output.mkdir(parents=True, exist_ok=False)
    data = torch.load(prepared / "episodes.pt", map_location="cpu", weights_only=True)
    episodes = [Episode(**e) for e in data[VARIANTS[variant]["dataset"]]]
    if not any(e.metadata["outcome"]["success"] for e in episodes):
        write_json(output / "skipped.json", {"reason": "No successful demonstration", "variant": variant, "seed": seed})
        return
    config, task = reference_setup(size=protocol["config"]["size"])
    adapter = matched_adapter(variant, seed, config.size).to(device)
    torch.manual_seed(seed + 100000)
    metrics = []

    def progress(event):
        metrics.append(event)
        write_json(output / "progress.json", event)
        if event["update"] % 500 == 0:
            print(variant, seed, event, flush=True)

    budget = protocol["updates"] if updates is None else updates
    started = time.perf_counter()
    losses = train(adapter, episodes, updates=budget, batch_size=protocol["batch_size"], seed=seed, progress=progress)
    seconds = time.perf_counter() - started
    save_checkpoint(output / "model.pt", adapter, config, task)
    windows = LeWAMWindows(episodes, adapter.history, adapter.model.chunk_size)
    write_json(
        output / "training.json",
        {
            "variant": variant,
            "seed": seed,
            "updates": budget,
            "training_seconds": seconds,
            "dataset_sha256": protocol["dataset_sha256"],
            "recipe": adapter.recipe,
            "parameters": sum(p.numel() for p in adapter.parameters()),
            "episodes": len(episodes),
            "successful_episodes": sum(e.metadata["outcome"]["success"] for e in episodes),
            "windows": len(windows),
            "losses": losses,
            "metrics": metrics,
            "torch_version": torch.__version__,
            "device": device,
            "initialization": "Equal-shaped tensors shared with same-seed H3; last three history positions shared",
        },
    )


def evaluate_variant(prepared, trained, *, device="cpu"):
    prepared, trained = Path(prepared), Path(trained)
    path = trained / "evaluation.json"
    if path.exists():
        raise FileExistsError("Evaluation already exists; preserve the recorded result")
    protocol = json.loads((prepared / "protocol.json").read_text())
    training = json.loads((trained / "training.json").read_text())
    if training["dataset_sha256"] != protocol["dataset_sha256"]:
        raise ValueError("Training and evaluation preparation differ")
    adapter, config, task, _ = restore_run(trained, device=device)
    data = torch.load(prepared / "episodes.pt", map_location="cpu", weights_only=True)
    demo = Episode(**data["clean"][0])
    report = {
        "variant": training["variant"],
        "seed": training["seed"],
        "trials": [],
        "tracking": {},
        "eligible_for_sim2real_gate": False,
        "physical_results": None,
        "tracking_reference": "Common clean timed-expert demonstration, not the observed expert's own targets",
    }
    for noise in protocol["noise_scales"]:
        report["tracking"][str(noise)] = action_tracking(adapter, demo, planner=LeWAMPlanner(noise_scale=noise))
    with SO101Chess(config, task, randomization=RandomizationConfig(enabled=False)) as env:
        for noise in protocol["noise_scales"]:
            for case in protocol["evaluation_cases"]:
                # Gaussian sampling needs repeats; centre initialization is deterministic.
                policy_seeds = protocol["policy_seeds"] if case["name"] == "nominal" and noise else [101]
                for policy_seed in policy_seeds:

                    def factory(goal, policy_seed=policy_seed, noise=noise):
                        return WorldModelPolicy(
                            adapter, goal, seed=policy_seed, planner=LeWAMPlanner(noise_scale=noise)
                        )

                    episode, result = trace_episode(
                        env,
                        **{**TASK, "seed": case["seed"]},
                        policy_factory=factory,
                        perturbations={int(k): v for k, v in case["perturbations"].items()},
                        include_commanded=True,
                    )
                    result.update(case=case["name"], policy_seed=policy_seed, noise_scale=noise)
                    report["trials"].append(result)
                    write_json(trained / "evaluation-progress.json", report)
                    if case["name"] == "nominal" and policy_seed == 101:
                        save_video(trained / f"nominal-noise-{noise}.gif", episode.pixels, control_dt=config.control_dt)
                    print(
                        training["variant"],
                        training["seed"],
                        noise,
                        case["name"],
                        policy_seed,
                        result["outcome"]["success"],
                        result["outcome"]["lifted"],
                        result["failure"],
                        flush=True,
                    )
    write_json(path, report)
    return report


def summarize(prepared, models, output):
    """Aggregate complete, matched runs; retain each trial and training seed.

    Repeated policy seeds share a scene and model, so no independent-trial
    confidence interval is asserted. The next-stage criterion is exploratory
    and cannot enable hardware or certify transfer to new moves.
    """
    prepared, models = Path(prepared), Path(models)
    protocol = json.loads((prepared / "protocol.json").read_text())
    report = {k: v for k, v in protocol.items() if k not in ("training", "controls")}
    report["training_attempts"] = {
        k: [{key: value for key, value in r.items() if key != "trace"} for r in group]
        for k, group in protocol["training"].items()
    }
    report["controls"] = [{k: v for k, v in r.items() if k != "trace"} for r in protocol["controls"]]
    report["runs"], report["summary"] = [], []
    for variant in protocol["variants"]:
        rows = []
        for seed in protocol["model_seeds"]:
            folder = models / f"{variant}-seed{seed}"
            training = json.loads((folder / "training.json").read_text())
            evaluation = json.loads((folder / "evaluation.json").read_text())
            if training["dataset_sha256"] != protocol["dataset_sha256"] or training["updates"] != protocol["updates"]:
                raise ValueError("Cannot aggregate different datasets or training budgets")
            expected = {
                (noise, case["name"], policy_seed)
                for noise in protocol["noise_scales"]
                for case in protocol["evaluation_cases"]
                for policy_seed in (protocol["policy_seeds"] if case["name"] == "nominal" and noise else [101])
            }
            actual = [(r["noise_scale"], r["case"], r["policy_seed"]) for r in evaluation["trials"]]
            if len(actual) != len(expected) or set(actual) != expected:
                raise ValueError("Incomplete or duplicated evaluation trials")
            if any(r["variant"] != variant or r["seed"] != seed for r in (training, evaluation)):
                raise ValueError("Training/evaluation identity differs from protocol")
            trials = [{k: v for k, v in r.items() if k != "trace"} for r in evaluation["trials"]]
            rows.extend(trials)
            report["runs"].append(
                {
                    "training": training,
                    "trials": trials,
                    "tracking": {
                        k: {a: b for a, b in v.items() if a != "actions"} for k, v in evaluation["tracking"].items()
                    },
                    "checkpoint": str(folder / "model.pt"),
                    "checkpoint_sha256": hashlib.sha256((folder / "model.pt").read_bytes()).hexdigest(),
                }
            )
        for noise in protocol["noise_scales"]:
            for domain in ("nominal", "disturbed"):
                selected = [
                    r for r in rows if r["noise_scale"] == noise and (r["case"] == "nominal") == (domain == "nominal")
                ]
                report["summary"].append(
                    {
                        "variant": variant,
                        "noise_scale": noise,
                        "domain": domain,
                        "trials": len(selected),
                        "successes": sum(r["outcome"]["success"] for r in selected),
                        "lifts": sum(r["outcome"]["lifted"] for r in selected),
                        "rejections": sum(r["failure"] is not None for r in selected),
                        "success_by_model_seed": {
                            str(run["training"]["seed"]): sum(
                                r["outcome"]["success"]
                                for r in run["trials"]
                                if r["noise_scale"] == noise and (r["case"] == "nominal") == (domain == "nominal")
                            )
                            for run in report["runs"]
                            if run["training"]["variant"] == variant
                        },
                    }
                )
    report["next_stage_candidates"] = [
        variant
        for variant in protocol["variants"]
        if all(
            r["successes"] / r["trials"] >= 0.9 and r["rejections"] == 0
            for r in report["summary"]
            if r["variant"] == variant and r["noise_scale"] == 0
        )
    ]
    report["next_stage_rule"] = (
        "Centre policy: >=90% success and no rejection in each domain; validate on new moves next"
    )
    report["complete"] = True
    write_json(output, report)
    return report


def run_curriculum(output, *, device="cpu"):
    """Execute the frozen five-variant, three-training-seed study in one process."""
    output = Path(output)
    prepare(output / "prepared")
    for seed in (17, 29, 43):
        for variant in VARIANTS:
            folder = output / "models" / f"{variant}-seed{seed}"
            train_variant(output / "prepared", folder, variant, seed, device=device)
            evaluate_variant(output / "prepared", folder, device=device)
    return summarize(output / "prepared", output / "models", output / "results.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("prepare")
    collect.add_argument("output", type=Path)
    collect.add_argument("--size", type=int, default=64)
    collect.add_argument("--recovery-count", type=int, default=12)
    fit = sub.add_parser("train")
    fit.add_argument("prepared", type=Path)
    fit.add_argument("output", type=Path)
    fit.add_argument("--variant", choices=VARIANTS, required=True)
    fit.add_argument("--seed", type=int, required=True)
    fit.add_argument("--updates", type=int)
    fit.add_argument("--device", default="cpu")
    test = sub.add_parser("evaluate")
    test.add_argument("prepared", type=Path)
    test.add_argument("trained", type=Path)
    test.add_argument("--device", default="cpu")
    aggregate = sub.add_parser("summarize")
    aggregate.add_argument("prepared", type=Path)
    aggregate.add_argument("models", type=Path)
    aggregate.add_argument("output", type=Path)
    full = sub.add_parser("run")
    full.add_argument("output", type=Path)
    full.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "prepare":
        prepare(args.output, size=args.size, recovery_count=args.recovery_count)
    elif args.command == "train":
        train_variant(args.prepared, args.output, args.variant, args.seed, device=args.device, updates=args.updates)
    elif args.command == "summarize":
        summarize(args.prepared, args.models, args.output)
    elif args.command == "run":
        run_curriculum(args.output, device=args.device)
    else:
        evaluate_variant(args.prepared, args.trained, device=args.device)


if __name__ == "__main__":
    main()
