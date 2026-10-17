"""Generate uniform notebook templates; pass --overwrite to replace saved runs."""

import argparse
import hashlib
import json
from importlib.resources import files
from pathlib import Path

from torch_world.envs import list_envs

LEGACY_NOTES = {
    "PointMass-v0": "GT is fixed at (0.75, 0.25), with the original seeded random starts and 0.1 success radius.",
    "Macaw-v0": (
        "GT is fixed at (0.75, 0.25). Presentation frames preserve the wing phase; GT shows the gliding pose. "
        "A single frame does not reveal wing phase, so the history=1 baseline is deliberately limited."
    ),
    "PushT-v0": (
        "This is a fixed symmetric translation from (0, 0.06, 0) to (0, -0.10, 0), pushed from above. "
        "It is one task instance, not the full random-pose Push-T distribution. "
        "Success retains the original position and angle thresholds."
    ),
    "RobotCube-v0": (
        "The UR10e uses a Robotiq 2F-85. GT is fixed at (0.69, -0.10) on the table. "
        "Actions are (dx, dy, dz, grip), with positive grip opening the fingers. "
        "Benchmark success also requires open fingers and a withdrawn tool. "
        "The overview shows the whole arm; model inputs keep the original task camera."
    ),
    "RobotCubeSO101-v0": (
        "The SO-101 uses its integrated gripper. GT is fixed at (0.209, -0.04) on the table. "
        "Actions are (dx, dy, dz, grip), with positive grip opening the fingers. "
        "Benchmark success also requires open fingers and a withdrawn tool. "
        "The overview shows the whole arm; model inputs keep the original task camera."
    ),
}


def cell(kind, source):
    out = {
        "cell_type": kind,
        "id": hashlib.sha256(source.encode()).hexdigest()[:8],
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        out.update(execution_count=None, outputs=[])
    return out


def main(overwrite=False):
    folder = Path(__file__).resolve().parent
    first = ["Maze-v0", "KeyDoor-v0", "Sokoban-v0"]
    specs = sorted(
        list_envs(),
        key=lambda s: (4 if "legacy" in s.tags else first.index(s.id) if s.id in first else 3, s.backend, s.id),
    )
    index = [
        "# Environment notebooks\n",
        "LeWM is the default; replace `model_factory` to compare another WorldModel.\n",
        (
            "Execution status is shown for every notebook. Executed outputs are small validation runs, "
            "not tuned benchmark scores.\n"
        ),
        "See [setup and protocol](https://github.com/latent-horizon/pytorch-world/blob/main/docs/guides/benchmark-environments.md).\n",
        "[Custom model across three environments](custom_model_three_envs.ipynb): compare a small residual world model with LeWM using one shared protocol.\n",
        "[Rendering quality comparison](render_quality_comparison.ipynb): three presets on Maze, BallPlate, and RGB FrankaCubeLift.\n",
        "| Notebook | Backend | Status | Task |",
        "|---|---|---|---|",
    ]
    if (folder / "RESULTS.md").exists():
        index.insert(4, "[Executed results and benchmark summary](RESULTS.md).\n")
    if (folder / "ROBOTIQ_RESULTS.md").exists():
        index.insert(5, "[Robotiq 2F-85 results for BlockStack and PickPlace](ROBOTIQ_RESULTS.md).\n")
    if (folder / "LEGACY_RESULTS.md").exists():
        index.insert(5, "[Original environment results: Macaw, PointMass, PushT and RobotCube](LEGACY_RESULTS.md).\n")
    for i, spec in enumerate(specs, 1):
        filename = f"{i:02d}_{spec.id.lower().replace('-', '_')}.ipynb"
        description = (
            f"# {spec.id}\n\n{spec.description}.\n\n"
            f"**{spec.backend}** · [Environment reference]({spec.reference}).\n\n"
            "A small LeWM baseline experiment: held-out dynamics prediction, a persistence control, "
            "and closed-loop planning against random actions. This validates the experiment pipeline; "
            "it is not a paper-scale training run."
        )
        if spec.backend == "isaaclab":
            description += (
                "\n\nRun in an Isaac Lab 3.x GPU runtime. This default uses policy state vectors and an MLP "
                'encoder, with actual simulator cameras for previews and episodes. Set `observation="rgb"` '
                'to train from camera images; the encoder switches to a CNN. `camera="overview"` gives a wider view. '
                "The green marker is the target. GT shows that marker for reach, or the object at its target for push/lift; "
                "the reference arm pose is not a solved manipulation trajectory."
            )
        class_name = "RobotCube" if spec.id == "RobotCubeSO101-v0" else spec.id.removesuffix("-v0")
        legacy = "legacy" in spec.tags
        setup = f"""from pathlib import Path

from torch_world.benchmarks import observation_encoder, run_experiment, show_results
from torch_world.envs import {class_name}
from torch_world.models import LeWM
from torch_world.planning import CEM
"""
        config = f"""world = {class_name}({('device="cuda:0", quality="high"' if spec.backend == "isaaclab" else 'quality="high"')})
SEED, UPDATES = 7, 400
OUTPUT_DIR = Path("runs/benchmarks")

# Replace this factory to compare another WorldModel with the same experiment.
def model_factory(observation_shape, action_dim):
    return LeWM(
        observation_encoder(observation_shape), action_dim, embed_dim=32, history=1,
        depth=2, num_heads=4, head_dim=8, mlp_dim=64, proj_dim=64, dropout=0,
    )


def planner_factory(action_dim):
    return CEM(8, action_dim, num_samples=96, num_elites=12, iterations=3, bounds=(-1, 1))


result = run_experiment(
    world, model_factory=model_factory, planner_factory=planner_factory,
    seed=SEED, updates=UPDATES, output_dir=OUTPUT_DIR,
)
"""
        settings, experiment = config.split("\n\nresult = run_experiment", 1)
        if legacy:
            arguments = 'size=128, quality="high"'
            if spec.id == "RobotCubeSO101-v0":
                arguments = 'robot="so101", ' + arguments
            settings = settings.replace(f'{class_name}(quality="high")', f"{class_name}({arguments})")
            description += (
                "\n\nThe named class works directly with the shared experiment and visualization API. "
                "Its original collection API remains available. GT is fixed throughout the experiment; "
                "training and evaluation use separate reset seeds. RobotCube assets are downloaded once into "
                "the existing Menagerie cache."
            )
            description += "\n\n" + LEGACY_NOTES.get(spec.id, "")
        experiment = "result = run_experiment" + experiment
        protocol = (
            "## Protocol\n\n2,048 requested training pairs; 400 updates; four evaluation seeds; 60-step "
            "evaluation budget. Puzzle training and holdout use disjoint starting states. Other tasks use "
            "separate collection seeds. Goal-renderable tasks use goal-image MPC; others learn a reward "
            "head on frozen embeddings and plan predicted rewards. Actual simulator returns are scored. "
            "The one-step model protocol uses `history=1`.\n\n"
            "Rendering defaults to high quality: 128-pixel model inputs and 512-pixel presentation frames. "
            'Use `quality="low"` or `quality="medium"` to compare cost and quality; `size` and `render_size` can be set independently.\n\n'
            "Fixed puzzle layouts use the same start for all four planner seeds. Success intervals therefore "
            "describe policy randomness on that level, not generalization across levels."
        )
        if legacy:
            protocol = protocol.split("\n\nFixed puzzle layouts")[0] + (
                "\n\nTraining uses random actions through the shared runner, not the older scripted play-data collectors. "
                "Model and random evaluation use matched starts and fixed GT. Seeds vary starts except in the fixed PushT task."
            )
        cells = [
            cell("markdown", description),
            cell("code", setup),
            cell("markdown", protocol),
            cell("code", settings),
            cell("code", "world.show_env()\n"),
            cell("code", experiment),
            cell(
                "markdown",
                "## GT reference\n\nA verified successful episode GIF with persistent GT, separate from measured model results. Reference controllers may use a longer episode budget; the GIF shows actual simulator steps.",
            ),
            cell("code", "world.show_success()\n"),
            cell("markdown", "## Results\n\nLower prediction error alone does not establish planning competence."),
            cell("code", 'show_results(result)\nprint("Saved run:", result["run_dir"])\n'),
            cell("code", "world.close()\n"),
        ]
        has_reference = (
            legacy
            or spec.backend in ("torch", "isaaclab")
            or files("torch_world.envs").joinpath("reference_actions", f"{spec.id}.json").is_file()
        )
        if not has_reference:
            cells[6] = cell(
                "markdown",
                "## Reference pending\n\nA verified default success controller is not yet available for this task. "
                "Supply your controller with `world.show_success(policy=your_controller)` to display an episode GIF "
                "with GT. Failed rollouts are rejected. The measured model results below remain available.",
            )
            del cells[7]
        path = folder / filename
        if overwrite or not path.exists():
            path.write_text(
                json.dumps(
                    {
                        "cells": cells,
                        "metadata": {
                            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                            "language_info": {"name": "python", "version": "3.12"},
                        },
                        "nbformat": 4,
                        "nbformat_minor": 5,
                    },
                    indent=1,
                )
                + "\n"
            )
        saved = json.loads(path.read_text())
        code = [c for c in saved["cells"] if c["cell_type"] == "code"]
        complete = code and all(c.get("execution_count") is not None for c in code)
        errors = any(o.get("output_type") == "error" for c in code for o in c.get("outputs", []))
        status = "Executed" if complete and not errors else "Incomplete" if errors else "Template"
        if not has_reference:
            status += "; reference pending"
        index.append(f"| [{spec.id}]({filename}) | {spec.backend} | {status} | {spec.description} |")
    index.extend(
        [
            "\n## Generate results\n",
            "Run all environment notebooks in fresh kernels with the shared LeWM/CEM protocol:",
            "```bash",
            "python examples/envs/run_notebooks.py --output-dir runs/benchmarks/my-campaign",
            "```",
            (
                "Pass `--backends torch mujoco` to select backends, or `--environments BlockStack-v0 PickPlace-v0` "
                "to select tasks. Use `--physics-python /path/to/isaaclab/.venv/bin/python` "
                "for Isaac Lab/Newton notebooks; that runtime also needs `ipykernel`. Output paths are relative to this directory."
            ),
            (
                "Repeat the same command to retry failed notebooks and skip completed ones. Each campaign contains "
                "`execution.json`, `summary.csv`, and versioned per-environment runs with metrics, provenance, GT and episodes. "
                "Every episode GIF includes GT. Provenance hashes include packaged hardware models and meshes. "
                "Keep different library hashes and task configurations separate when comparing campaigns."
            ),
        ]
    )
    (folder / "README.md").write_text("\n".join(index) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args().overwrite)
