"""Reproducible simulation-only CLI: python -m torch_world.sim2real.run."""

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from torch_world.planning import CEM, LeWAMPlanner
from torch_world.sim2real.config import RandomizationConfig
from torch_world.sim2real.evaluation import evaluate_saved_run, restore_run
from torch_world.sim2real.experiment import run_experiment
from torch_world.sim2real.reference import reference_setup, simulation_gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("dinowm", "lewm", "lewam", "robojepa"), default="dinowm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=140)
    parser.add_argument("--episodes", type=int, default=72)
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("runs/so101-reference"))
    parser.add_argument("--checkpoint-run", type=Path, help="Evaluate saved weights/data without training again")
    parser.add_argument("--policy", choices=("direct", "ranked", "cem"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_per_process_memory_fraction(0.08)
    config, task = reference_setup(size=args.size, seed=args.seed)
    if args.checkpoint_run is not None:
        restored, config, task, _ = restore_run(args.checkpoint_run, device="cpu")
        args.model = restored.kind
    args.policy = args.policy or ("direct" if args.model == "lewam" else "cem")
    if args.model != "lewam" and args.policy != "cem":
        parser.error("direct/ranked action policies require --model lewam")
    limits = torch.tensor([1, 1, 1, config.max_gripper_step], device=args.device)

    def planner():
        if args.model == "lewam" and args.policy != "cem":
            return LeWAMPlanner(num_samples=1 if args.policy == "direct" else args.samples)
        return CEM(
            args.horizon,
            4,
            num_samples=args.samples,
            num_elites=max(2, args.samples // 8),
            iterations=args.iterations,
            bounds=(-limits, limits),
        )

    started = time.time()

    def progress(event):
        # Compact logs deliberately omit image arrays and long timestamp lists.
        keep = ("stage", "domain", "policy", "episode", "total", "update", "loss", "success", "failure")
        print(
            json.dumps(
                {"elapsed_seconds": round(time.time() - started, 1), **{k: event[k] for k in keep if k in event}}
            ),
            flush=True,
        )

    print(
        json.dumps({"host": platform.node(), "torch": torch.__version__, "arguments": vars(args)}, default=str),
        flush=True,
    )
    if args.checkpoint_run is not None:
        result = evaluate_saved_run(
            args.checkpoint_run,
            randomization=RandomizationConfig(),
            output_dir=args.output,
            seeds=tuple(range(10007, 10007 + args.eval_episodes)),
            planner_factory=planner,
            device=args.device,
            progress=progress,
        )
    else:
        result = run_experiment(
            config=config,
            task=task,
            model=args.model,
            smoke=False,
            device=args.device,
            episodes=args.episodes,
            updates=args.updates,
            batch_size=args.batch_size,
            cache_features=args.model == "dinowm",
            evaluation_seeds=tuple(range(10007, 10007 + args.eval_episodes)),
            output_dir=args.output,
            planner_factory=planner,
            progress=progress,
        )
    folder = Path(result["run_dir"])
    report = {
        "host": platform.node(),
        "torch": torch.__version__,
        "arguments": vars(args),
        "elapsed_seconds": time.time() - started,
        "gate": simulation_gate(result),
    }
    (folder / "execution.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"run_dir": str(folder), **report}, default=str), flush=True)


if __name__ == "__main__":
    main()
