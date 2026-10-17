"""Compare a fresh model on an existing simulation dataset and evaluation tasks."""

import argparse
import json
import platform
import shutil
import time
from pathlib import Path

import torch

from torch_world.planning import CEM, LeWAMPlanner
from torch_world.sim2real.config import RandomizationConfig
from torch_world.sim2real.data import Episode
from torch_world.sim2real.evaluation import evaluate_saved_run, restore_run
from torch_world.sim2real.experiment import _folder, _json, save_checkpoint, train
from torch_world.sim2real.models import make_adapter
from torch_world.sim2real.reference import simulation_gate


def retrain_saved_dataset(
    source,
    *,
    model="lewm",
    updates=3000,
    batch_size=16,
    device="cuda",
    output_dir,
    progress=None,
    proprioception=False,
    rollout_steps=1,
    chunk_size=8,
):
    """Train from scratch on exactly the source run's train split; preserve holdout.

    Evaluation is separate so callers can reproduce the source planner and seeds.
    A copied dataset makes the new training artifact independent of the source.
    """
    source = Path(source).resolve()
    previous, config, task, calibration = restore_run(source, device="cpu")
    del previous
    torch.manual_seed(config.seed)
    adapter = make_adapter(
        model,
        size=config.size,
        proprioception=proprioception,
        rollout_steps=rollout_steps,
        chunk_size=chunk_size,
        action_scale=(1, 1, 1, config.max_gripper_step),
    ).to(device)
    data = torch.load(source / "episodes.pt", map_location="cpu", weights_only=True, mmap=True)
    episodes = [Episode(**episode) for episode in data["train"]]
    folder = _folder(output_dir)
    training_metrics = []

    def report(event):
        if event.get("stage") == "training":
            training_metrics.append(event)
        if progress is not None:
            progress(event)

    losses = train(
        adapter,
        episodes,
        updates=updates,
        batch_size=batch_size,
        seed=config.seed,
        cache_features=model == "dinowm",
        progress=report,
    )
    save_checkpoint(folder / "model.pt", adapter, config, task, calibration)
    _json(folder / "losses.json", losses)
    _json(folder / "training-metrics.json", training_metrics)
    shutil.copyfile(source / "episodes.pt", folder / "episodes.pt")
    _json(
        folder / "dataset-source.json",
        {
            "source_run": str(source),
            "splits": "unchanged",
            "updates": updates,
            "batch_size": batch_size,
            "seed": config.seed,
            "proprioception": proprioception,
            "rollout_steps": rollout_steps,
            "chunk_size": chunk_size if model == "lewam" else None,
            "lewam_protocol": "aligned applied transitions; successful terminal goals supervise actions; all episodes train dynamics"
            if model == "lewam"
            else None,
        },
    )
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--model", choices=("lewm", "lewam", "dinowm", "robojepa"), default="lewm")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--proprioception", action="store_true")
    parser.add_argument("--rollout-steps", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--policy", choices=("direct", "ranked", "cem"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = json.loads((args.source / "results.json").read_text())
    execution = json.loads((args.source / "execution.json").read_text())
    settings = execution["arguments"]
    seeds = tuple(e["seed"] for e in source["evaluation"]["nominal"]["learned"]["episodes"])
    randomized_seeds = tuple(e["seed"] for e in source["evaluation"]["randomized"]["learned"]["episodes"])
    if seeds != randomized_seeds:
        raise ValueError("Source evaluation domains must have the same seeds")
    args.policy = args.policy or ("direct" if args.model == "lewam" else "cem")
    if args.model != "lewam" and args.policy != "cem":
        parser.error("direct/ranked action policies require --model lewam")
    limits = torch.tensor([1, 1, 1, source["config"]["max_gripper_step"]], device=args.device)

    def planner():
        if args.model == "lewam" and args.policy != "cem":
            return LeWAMPlanner(num_samples=1 if args.policy == "direct" else settings["samples"])
        return CEM(
            settings["horizon"],
            4,
            num_samples=settings["samples"],
            num_elites=max(2, settings["samples"] // 8),
            iterations=settings["iterations"],
            bounds=(-limits, limits),
        )

    started = time.time()

    def progress(event):
        keep = (
            "stage",
            "domain",
            "policy",
            "episode",
            "total",
            "update",
            "loss",
            "pred_loss",
            "action_loss",
            "state_loss",
            "rollout_loss",
            "sigreg_loss",
            "success",
            "failure",
        )
        print(
            json.dumps(
                {"elapsed_seconds": round(time.time() - started, 1), **{k: event[k] for k in keep if k in event}}
            ),
            flush=True,
        )

    training = retrain_saved_dataset(
        args.source,
        model=args.model,
        updates=args.updates,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output / "training",
        progress=progress,
        proprioception=args.proprioception,
        rollout_steps=args.rollout_steps,
        chunk_size=args.chunk_size,
    )
    result = evaluate_saved_run(
        training,
        randomization=RandomizationConfig(**source["randomization"]),
        output_dir=args.output / "evaluation",
        seeds=seeds,
        planner_factory=planner,
        device=args.device,
        progress=progress,
    )
    report = {
        "host": platform.node(),
        "torch": torch.__version__,
        "source_run": str(args.source.resolve()),
        "training_run": str(training.resolve()),
        "arguments": vars(args),
        "planner": (
            {
                "type": "LeWAMPlanner",
                "policy": args.policy,
                "chunk_size": args.chunk_size,
                "num_samples": 1 if args.policy == "direct" else settings["samples"],
            }
            if args.model == "lewam" and args.policy != "cem"
            else {k: settings[k] for k in ("horizon", "samples", "iterations")}
        ),
        "elapsed_seconds": time.time() - started,
        "gate": simulation_gate(result),
    }
    report = json.loads(json.dumps(report, default=str))
    _json(Path(result["run_dir"]) / "execution.json", report)
    print(json.dumps({"run_dir": result["run_dir"], **report}), flush=True)


if __name__ == "__main__":
    main()
