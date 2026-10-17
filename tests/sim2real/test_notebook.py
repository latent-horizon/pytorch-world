from pathlib import Path

import pytest

pytest.importorskip("mujoco")
nbformat = pytest.importorskip("nbformat")
NotebookClient = pytest.importorskip("nbclient").NotebookClient


def test_notebook_executes_without_hardware(tmp_path):
    path = Path(__file__).resolve().parents[2] / "examples/sim2real/so101_chess.ipynb"
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(
        notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}
    ).execute()
    output = "\n".join(o.get("text", "") for cell in notebook.cells for o in cell.get("outputs", []))
    assert "Motor writes: 0" in output
    assert "Physical result: not measured" in output
    assert list((tmp_path / "runs/so101-chess").glob("*/manifest.json"))
