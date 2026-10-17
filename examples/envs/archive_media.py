"""Save embedded GIFs locally while keeping static notebook previews.

Run after executing notebooks: python examples/envs/archive_media.py
Local GIFs are gitignored. Recorded metrics are preserved; notebook diffs stay small enough
for repository viewers. Re-executing a notebook still produces inline animations.
"""

import base64
import hashlib
import io
import re
from pathlib import Path

import nbformat
from PIL import Image


def archive_media(path):
    path = Path(path)
    notebook = nbformat.read(path, as_version=4)
    changed = False
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        outputs = []
        for output in cell.outputs:
            html = output.get("data", {}).get("text/html", "")
            match = re.search(r'src=["\']data:image/gif;base64,([^"\']+)', html)
            if match is None:
                outputs.append(output)
                continue
            raw = base64.b64decode(match[1], validate=True)
            digest = hashlib.sha256(raw).hexdigest()
            relative = Path("media") / f"{digest}.gif"
            destination = path.parent / relative
            destination.parent.mkdir(exist_ok=True)
            if destination.exists() and destination.read_bytes() != raw:
                raise ValueError(f"Media checksum mismatch: {destination}")
            destination.write_bytes(raw)
            with Image.open(io.BytesIO(raw)) as gif:
                preview = io.BytesIO()
                gif.convert("RGB").save(preview, format="PNG")
            outputs.extend(
                [
                    nbformat.v4.new_output(
                        "display_data",
                        data={"image/png": base64.b64encode(preview.getvalue()).decode()},
                        metadata={"recorded_gif_sha256": digest},
                    ),
                    nbformat.v4.new_output(
                        "display_data",
                        data={
                            "text/markdown": "Static preview of the recorded episode. Rerun this cell to view the animation."
                        },
                    ),
                ]
            )
            changed = True
        cell.outputs = outputs
    if changed:
        nbformat.write(notebook, path)
    return changed


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    changed = sum(archive_media(path) for path in sorted(folder.glob("*.ipynb")))
    print(f"Archived media in {changed} notebooks")
