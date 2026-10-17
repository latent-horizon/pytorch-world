"""A campaign retries failures, keeps completed notebooks, and uses its selected runtime."""

import json
import runpy
import sys
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")


def test_notebook_campaign_retries_and_resumes(tmp_path):
    runner = runpy.run_path(str(Path(__file__).parents[1] / "examples/envs/run_notebooks.py"))["run_notebooks"]
    first, second = tmp_path / "01_maze.ipynb", tmp_path / "02_keydoor.ipynb"
    for path, task, code in (
        (first, "Maze-v0", 'raise RuntimeError("intentional test failure")'),
        (second, "KeyDoor-v0", f'import sys\nassert sys.prefix == {sys.prefix!r}\nprint("complete")'),
    ):
        notebook = nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_markdown_cell(f"# {task}"),
                nbformat.v4.new_code_cell(code),
            ]
        )
        nbformat.write(notebook, path)
    settings = {
        "folder": tmp_path,
        "output_dir": Path("runs/test"),
        "backends": ["torch"],
        "python": sys.executable,
        "timeout": 30,
    }
    unselected = first.read_text()
    assert runner(**settings, environments=["KeyDoor-v0"]) == []
    assert first.read_text() == unselected
    assert runner(**settings) == [first.name]
    complete = second.read_text()
    notebook = nbformat.read(first, as_version=4)
    notebook.cells[1].source = 'print("recovered")'
    nbformat.write(notebook, first)
    assert runner(**settings) == []
    assert second.read_text() == complete
    report = json.loads((tmp_path / "runs/test/execution.json").read_text())
    assert all(entry["status"] == "complete" for entry in report.values())


def test_media_archival_preserves_animation_metrics_and_is_idempotent(tmp_path):
    import base64
    import io

    image = pytest.importorskip("PIL.Image")
    archive = runpy.run_path(str(Path(__file__).parents[1] / "examples/envs/archive_media.py"))["archive_media"]
    stream = io.BytesIO()
    image.new("RGB", (8, 8), "red").save(
        stream, format="GIF", save_all=True, append_images=[image.new("RGB", (8, 8), "blue")], duration=100
    )
    raw = stream.getvalue()
    animation = nbformat.v4.new_output(
        "display_data", data={"text/html": f'<img src="data:image/gif;base64,{base64.b64encode(raw).decode()}"/>'}
    )
    score = nbformat.v4.new_output("display_data", data={"text/markdown": "Measured success: 0/4"})
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("show_results(result)", outputs=[score, animation])]
    )
    path = tmp_path / "experiment.ipynb"
    nbformat.write(notebook, path)
    assert archive(path)
    outputs = nbformat.read(path, as_version=4).cells[0].outputs
    assert outputs[0] == score and "image/png" in outputs[1].data
    gif = next((tmp_path / "media").glob("*.gif"))
    assert gif.read_bytes() == raw
    assert "Rerun this cell" in outputs[2].data["text/markdown"]
    assert "media/" not in outputs[2].data["text/markdown"]
    saved = path.read_bytes()
    assert not archive(path) and saved == path.read_bytes()
