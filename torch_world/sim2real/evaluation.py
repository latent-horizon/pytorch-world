"""Evaluate a saved simulation training run without collecting or training again."""

import json
from dataclasses import asdict
from pathlib import Path

import torch

from torch_world.sim2real.calibration import Calibration
from torch_world.sim2real.config import BoardAppearance, ChessTaskConfig, PieceSpec, SO101Config, WristView, fingerprint
from torch_world.sim2real.data import Episode
from torch_world.sim2real.experiment import _folder, _json, evaluate, heldout_loss, load_checkpoint, save_video


def restore_run(training_run, *, device="cpu"):
    """Restore the exact saved camera/task contract, including nested settings."""
    path = Path(training_run) / "model.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    control, geometry = dict(saved["config"]), dict(saved["task"])
    control["wrist_view"] = WristView(**control["wrist_view"])
    geometry["appearance"] = BoardAppearance(**geometry["appearance"])
    geometry["pieces"] = {name: PieceSpec(**value) for name, value in geometry["pieces"].items()}
    config, task = SO101Config(**control), ChessTaskConfig(**geometry)
    calibration = None if saved["calibration"] is None else Calibration(**saved["calibration"])
    adapter = load_checkpoint(path, config, task, calibration, device=device)
    return adapter, config, task, calibration


def evaluate_saved_run(
    training_run, *, randomization, output_dir, seeds, planner_factory=None, device="cpu", steps=None, progress=None
):
    """Write a separate evaluation, preserving original weights and training data.

    ``randomization`` is explicit: an evaluation never silently invents the
    domain ranges of an older checkpoint. Seeds must not overlap saved data.
    """
    training_run = Path(training_run).resolve()
    adapter, config, task, calibration = restore_run(training_run, device=device)
    data = torch.load(training_run / "episodes.pt", map_location="cpu", weights_only=True)
    used_seeds = {e["metadata"]["seed"] for group in data.values() for e in group}
    if not seeds or used_seeds.intersection(seeds):
        raise ValueError("Evaluation seeds must be nonempty and disjoint from training/validation")
    folder = _folder(output_dir)
    losses = json.loads((training_run / "losses.json").read_text())
    result = {
        "schema": 1,
        "model": adapter.recipe,
        "config": asdict(config),
        "task": asdict(task),
        "randomization": asdict(randomization),
        "calibration_id": None if calibration is None else calibration.id,
        "training_episodes": [e["metadata"] for e in data["train"]],
        "holdout_episodes": [e["metadata"] for e in data["holdout"]],
        "losses": losses,
        "updates": len(losses),
        "smoke": adapter.recipe["smoke"],
        "physical_results": None,
        "training_run": str(training_run),
        "evaluation": {},
    }
    # Free large training image tensors before validation and planning.
    holdout = [Episode(**e) for e in data["holdout"]]
    del data
    result["heldout_loss"] = heldout_loss(adapter, holdout)
    del holdout
    videos = {}
    for name, setting in (("nominal", type(randomization)(enabled=False)), ("randomized", randomization)):

        def report(event, domain=name):
            event = {**event, "domain": domain}
            _json(folder / "progress.json", event)
            # Preserve every completed trial if a native renderer later stalls.
            with (folder / "episodes.jsonl").open("a") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            if progress is not None:
                progress(event)

        metrics, frames = evaluate(
            adapter,
            config,
            task,
            setting,
            seeds=seeds,
            steps=steps,
            calibration=calibration,
            planner_factory=planner_factory,
            progress=report,
        )
        result["evaluation"][name], videos[name] = metrics, frames
        _json(folder / "results.partial.json", result)
        for policy, sequence in frames.items():
            save_video(folder / f"{name}-{policy}.gif", sequence, control_dt=config.control_dt)
    _json(folder / "results.json", result)
    torch.save(videos, folder / "videos.pt")
    _json(
        folder / "manifest.json",
        {
            "schema": 1,
            "complete": True,
            "contract": fingerprint(config, task, calibration),
            "training_run": str(training_run),
        },
    )
    return {**result, "run_dir": str(folder)}
