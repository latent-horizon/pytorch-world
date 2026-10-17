"""Replay verified reference episodes and refresh only their notebook output cells.

Run one simulator per process, e.g. ``python examples/envs/render_references.py Maze-v0``.
Experiment results are preserved. Reference GIFs and provenance live in their own
campaign directory, separate from model evaluation artifacts.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

from torch_world.benchmarks.artifacts import library_fingerprint
from torch_world.envs import get_spec, make


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment")
    parser.add_argument("--quality", choices=("low", "medium", "high"), default="high")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/references"))
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    spec = get_spec(args.environment)
    campaign = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = folder / args.output_dir / spec.id / campaign
    output.mkdir(parents=True, exist_ok=False)
    InteractiveShell.instance()
    with make(spec.id, quality=args.quality) as world:
        with capture_output() as captured:
            world.show_success(seed=7, path=output / "episode.gif")
        record = {
            "environment": spec.id,
            "seed": 7,
            "success": bool(world.transition.success[0]),
            "steps": int(world.elapsed[0]),
            "render_settings": world.render_settings,
            "kind": "reference_episode",
            "gif_sha256": hashlib.sha256((output / "episode.gif").read_bytes()).hexdigest(),
        }
    packages = {}
    for name in ("torch", "mujoco", "newton", "isaaclab"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    record.update(packages=packages, library_source_sha256=library_fingerprint())
    (output / "reference.json").write_text(json.dumps(record, indent=2) + "\n")
    paths = list(folder.glob(f"[0-9][0-9]_{spec.id.lower().replace('-', '_')}.ipynb"))
    if len(paths) != 1:
        raise ValueError(f"Expected one notebook for {spec.id}, found {len(paths)}")
    notebook = nbformat.read(paths[0], as_version=4)
    next_execution = max((cell.get("execution_count") or 0 for cell in notebook.cells), default=0) + 1
    for cell in notebook.cells:
        if cell.cell_type == "markdown" and cell.source.startswith("## GT reference"):
            cell.source = (
                "## GT reference\n\nA verified successful episode GIF, separate from the model results. "
                "The GT panel remains visible throughout. Reference controllers may use a longer "
                "episode budget than model evaluation; the GIF shows actual simulator step numbers."
            )
        if cell.cell_type == "code" and cell.source.strip() == "world.show_success()":
            cell.execution_count = next_execution
            cell.outputs = [
                nbformat.v4.new_output("display_data", data=item.data, metadata=item.metadata)
                for item in captured.outputs
            ]
            cell.metadata["reference"] = record
    nbformat.write(notebook, paths[0])
    print(json.dumps({**record, "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
