"""Versioned, append-only benchmark artifacts and cross-run summaries."""

import csv
import hashlib
import json
import re
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

SCHEMA_VERSION = 1


def library_fingerprint():
    """Hash source, reference actions and packaged hardware geometry in path order."""
    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for source in sorted(package_root.rglob("*")):
        if source.suffix in {".py", ".json", ".xml", ".stl"}:
            digest.update(str(source.relative_to(package_root)).encode())
            digest.update(source.read_bytes())
    return digest.hexdigest()


def _slug(value):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._") or "unnamed"


def configuration(value):
    """Convert recipe settings into JSON values without serializing live simulator objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): configuration(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [configuration(v) for v in value]
    if isinstance(value, set):
        return sorted(configuration(v) for v in value)
    if is_dataclass(value) and not isinstance(value, type):
        return configuration(asdict(value))
    if hasattr(value, "tolist"):
        return configuration(value.tolist())
    if hasattr(value, "to_dict"):
        return configuration(value.to_dict())
    if isinstance(value, partial):
        return {
            "factory": configuration(value.func),
            "args": configuration(value.args),
            "kwargs": configuration(value.keywords),
        }
    if callable(value):
        return f"{value.__module__}:{value.__qualname__}"
    if isinstance(value, (torch.device, Path)):
        return str(value)
    raise TypeError(f"Cannot record {type(value).__name__}; provide JSON-serializable experiment settings")


def save_run(result, root, *, gifs=True):
    """Save one immutable run and return its directory; never overwrite another run.

    ``manifest.json`` is written last, marking a complete run. Aggregation ignores
    interrupted directories. JSON contains only scalar/list metadata; episode
    tensors are separate ``.pt`` files loadable with ``weights_only=True``.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{stamp}-{uuid.uuid4().hex[:8]}"
    folder = Path(root) / _slug(result["env_id"]) / _slug(result["model"]) / run_id
    folder.mkdir(parents=True, exist_ok=False)
    config = configuration(result["config"])
    config_text = json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n"
    (folder / "config.json").write_text(config_text)
    metrics = {
        k: v
        for k, v in result.items()
        if k not in ("config", "losses", "run_dir", "target") and not k.endswith(" frames")
    }
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n")
    with (folder / "training.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["update", "loss"])
        writer.writerows(enumerate(result["losses"], 1))
    episodes = {k.removesuffix(" frames"): v.cpu() for k, v in result.items() if k.endswith(" frames")}
    if episodes:
        torch.save(episodes, folder / "episodes.pt")
    if "target" in result:
        torch.save(result["target"].cpu(), folder / "target.pt")
    if gifs:
        from torch_world.benchmarks.visualization import episode_gif

        for policy, frames in episodes.items():
            if frames.ndim == 4 and frames.shape[1] == 3:
                successes = result["evaluation"].get(policy, {}).get("successes")
                success = None if successes is None else bool(successes[0])
                episode_gif(
                    frames,
                    result["target"],
                    title=policy,
                    success=success,
                    fps=4,
                    stride=2,
                    target_title=result.get("gt_label", "GT"),
                ).save(folder / f"{_slug(policy)}.gif")
    packages = {}
    for name in (
        "torch-world",
        "torch",
        "mujoco",
        "gymnasium",
        "gymnasium-robotics",
        "newton",
        "warp-lang",
        "isaaclab",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    files = sorted(p for p in folder.iterdir() if p.is_file())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "library_source_sha256": library_fingerprint(),
        "run_id": run_id,
        "created_utc": stamp,
        "environment": result["env_id"],
        "model": result["model"],
        "seed": result["seed"],
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "packages": packages,
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return folder


def summarize_runs(root):
    """One row per completed run/policy; do not pool incompatible environments or metrics.

    Repeated seeds remain separate rows. Consumers choose aggregation by
    environment, model, library source hash, protocol/config hash and seed,
    avoiding accidental pooling across hardware or task revisions.
    """
    rows = []
    for path in sorted(Path(root).rglob("manifest.json")):
        manifest = _verify_manifest(path)
        metrics = json.loads((path.parent / "metrics.json").read_text())
        config = json.loads((path.parent / "config.json").read_text())
        render = config.get("render_settings", {})
        for policy, scores in metrics["evaluation"].items():
            rows.append(
                {
                    "run_id": manifest["run_id"],
                    "environment": manifest["environment"],
                    "model": manifest["model"],
                    "policy": policy,
                    "seed": manifest["seed"],
                    "config_sha256": manifest["config_sha256"],
                    "library_source_sha256": manifest["library_source_sha256"],
                    "observation": metrics["observation"],
                    "quality": render.get("quality"),
                    "observation_size": render.get("observation_size"),
                    "render_size": render.get("render_size"),
                    "camera": config.get("environment_settings", {}).get("camera"),
                    "success_rate": scores.get("success_rate"),
                    "mean_return": scores.get("mean_return"),
                    "mean_final_distance": (
                        sum(scores["final_distances"]) / len(scores["final_distances"])
                        if "final_distances" in scores
                        else None
                    ),
                    "heldout_latent_mse": metrics["heldout_latent_mse"],
                    "persistence_latent_mse": metrics["persistence_latent_mse"],
                    "run_dir": str(path.parent),
                }
            )
    return rows


def write_summary(root, destination):
    """Write a tidy CSV of completed run/policy rows; return the rows as Python dicts."""
    rows = summarize_runs(root)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def load_run(folder):
    """Load a completed, checksum-verified run for ``show_results`` without rerunning it."""
    folder = Path(folder)
    _verify_manifest(folder / "manifest.json")
    result = json.loads((folder / "metrics.json").read_text())
    result["config"] = json.loads((folder / "config.json").read_text())
    with (folder / "training.csv").open(newline="") as stream:
        result["losses"] = [float(row["loss"]) for row in csv.DictReader(stream)]
    if (folder / "episodes.pt").exists():
        episodes = torch.load(folder / "episodes.pt", map_location="cpu", weights_only=True)
        result.update({name + " frames": frames for name, frames in episodes.items()})
    if (folder / "target.pt").exists():
        result["target"] = torch.load(folder / "target.pt", map_location="cpu", weights_only=True)
    result["run_dir"] = str(folder)
    return result


def _verify_manifest(path):
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported artifact schema in {path}")
    for name, digest in manifest["files"].items():
        target = path.parent / name
        if target.parent != path.parent or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Artifact checksum mismatch: {target}")
    return manifest
