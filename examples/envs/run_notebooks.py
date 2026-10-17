"""Execute environment notebooks in fresh kernels and save a resumable campaign.

Run from the repository root. Physics notebooks can use a separate Python runtime
with Isaac Lab/Newton installed; every kernel needs the notebooks extra.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

from torch_world.benchmarks import write_summary
from torch_world.envs import get_spec

logger = logging.getLogger(__name__)


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _checkpoint(notebook, path, entry, report, report_path, cell, cell_index, **kwargs):
    entry["last_cell"] = cell_index
    nbformat.write(notebook, path)
    _write_json(report_path, report)
    print(f"  {path.name}: cell {cell_index} finished", flush=True)


def run_notebooks(folder, output_dir, *, backends, python, physics_python=None, timeout=1800, environments=None):
    """Run matching environment notebooks; repeat with the same output_dir to resume."""
    folder = Path(folder).resolve()
    output_dir = Path(output_dir)
    root = (folder / output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "execution.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    source_root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [str(source_root), os.environ.get("PYTHONPATH")])),
        "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl" if sys.platform == "linux" else "glfw"),
    }
    failures = []
    selected = {get_spec(name).id for name in environments} if environments is not None else None
    for path in sorted(folder.glob("[0-9][0-9]_*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        spec = get_spec(notebook.cells[0].source.splitlines()[0].removeprefix("# "))
        if spec.backend not in backends:
            continue
        if selected is not None and spec.id not in selected:
            continue
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.source = re.sub(
                    r"^OUTPUT_DIR = .+$", f"OUTPUT_DIR = Path({str(output_dir)!r})", cell.source, flags=re.MULTILINE
                )
        source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
        digest = hashlib.sha256(source.encode()).hexdigest()
        previous = report.get(path.name, {})
        if previous.get("status") == "complete" and previous.get("notebook_source_sha256") == digest:
            print(f"SKIP {path.name}: already complete", flush=True)
            continue
        interpreter = physics_python if spec.backend in ("isaaclab", "newton") and physics_python else python
        entry = report[path.name] = {
            "environment": spec.id,
            "backend": spec.backend,
            "python": str(Path(interpreter).expanduser().absolute()),
            "notebook_source_sha256": digest,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running",
        }
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs, cell.execution_count = [], None
                cell.metadata.pop("reference", None)
        print(f"RUN {path.name} ({spec.backend})", flush=True)
        _write_json(report_path, report)

        checkpoint = partial(_checkpoint, notebook, path, entry, report, report_path)

        try:
            with tempfile.TemporaryDirectory(prefix="torch-world-kernel-") as directory:
                kernel = Path(directory) / "python3"
                kernel.mkdir()
                _write_json(
                    kernel / "kernel.json",
                    {
                        "argv": [entry["python"], "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                        "display_name": "Python 3",
                        "language": "python",
                    },
                )
                manager = KernelManager(
                    kernel_name="python3", kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory])
                )
                client = NotebookClient(
                    notebook,
                    km=manager,
                    timeout=timeout,
                    kernel_name="python3",
                    resources={"metadata": {"path": str(folder)}},
                    on_cell_executed=checkpoint,
                )
                client.execute(cleanup_kc=True, env=env)
            entry["status"] = "complete"
            print(f"DONE {path.name}", flush=True)
        except Exception as error:
            logger.exception("Notebook failed: %s", path.name)
            entry.update(status="failed", error=f"{type(error).__name__}: {error}")
            failures.append(path.name)
            print(f"FAIL {path.name}: {type(error).__name__}", flush=True)
        finally:
            entry["finished_utc"] = datetime.now(timezone.utc).isoformat()
            notebook.metadata["benchmark_execution"] = dict(entry)
            nbformat.write(notebook, path)
            _write_json(report_path, report)
            write_summary(root, root / "summary.csv")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("torch", "mujoco", "newton", "isaaclab"),
        default=["torch", "mujoco", "newton", "isaaclab"],
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--environments", nargs="+", help="Run only these registered environment IDs")
    parser.add_argument("--physics-python", help="Python executable with Isaac Lab/Newton and ipykernel installed")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-cell timeout in seconds")
    args = parser.parse_args()
    failures = run_notebooks(**vars(args))
    if failures:
        raise SystemExit("Failed notebooks (rerun the same campaign to retry): " + ", ".join(failures))


if __name__ == "__main__":
    main()
