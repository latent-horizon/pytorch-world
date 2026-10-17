"""Build the catalog's tables and figures from a completed notebook campaign.

Usage: python examples/envs/report_results.py CAMPAIGN --output examples/envs/RESULTS.md
One completed run per environment is required; different campaigns are not pooled.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch

from torch_world import metrics, plots
from torch_world.benchmarks import write_summary

BACKENDS = {
    "torch": "PyTorch environments",
    "mujoco": "MuJoCo manipulation and control",
    "newton": "Newton control",
    "isaaclab": "Isaac Lab Franka",
}


def report_results(campaign, output, *, hardware=None):
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    rows = write_summary(campaign, campaign / "summary.csv")  # Validates every manifest/checksum.
    directories = sorted({Path(row["run_dir"]) for row in rows})
    if not directories:
        raise ValueError("No completed runs found")
    grouped, seen = defaultdict(list), set()
    for directory in directories:
        result = json.loads((directory / "metrics.json").read_text())
        config = json.loads((directory / "config.json").read_text())
        if result["env_id"] in seen:
            raise ValueError("Select a campaign with exactly one completed run per environment")
        seen.add(result["env_id"])
        grouped[config["backend"]].append((directory, result, config))

    def link(path):
        return Path(os.path.relpath(path, output.parent)).as_posix()

    notebooks = {}
    for path in Path(__file__).parent.glob("[0-9][0-9]_*.ipynb"):
        title = "".join(json.loads(path.read_text())["cells"][0]["source"]).splitlines()[0]
        notebooks[title.removeprefix("# ")] = path.resolve()
    lines = [
        "# Executed environment results\n",
        (
            f"{len(directories)} environments executed with the recorded model and planner configurations. "
            + (f"Hardware: {hardware}. " if hardware else "")
            + "These are validation runs, not tuned benchmark scores. Each notebook contains the exact training and evaluation budget.\n"
        ),
        (
            f"[Policy rows (CSV)]({link(campaign / 'summary.csv')}) · "
            f"[Execution record]({link(campaign / 'execution.json')}) · [Notebook index](README.md)\n"
        ),
        (
            "The high-quality preset is the default: 128-pixel RGB model inputs and 512-pixel presentation frames. "
            "Policy observation types are listed by backend; camera episodes are available independently. "
            "[The quality comparison](render_quality_comparison.ipynb) also evaluates RGB Franka models.\n"
        ),
        (
            "Every episode visualization includes GT. Franka reach GT identifies the green target marker; "
            "push/lift GT places the object at its required position. These arm poses are not successful policy demonstrations. "
            "Newton link images remain schematic.\n"
        ),
        (
            "Compare returns within an environment because reward scales differ. Charts show 95% Wilson intervals over "
            "the recorded evaluation episodes. Puzzle seeds vary policy randomness on one fixed layout. Goal tasks score "
            "final-state success; Franka scores success reached during the episode. Only the first seed is visualized.\n"
        ),
    ]
    figures = campaign / "figures"
    figures.mkdir(exist_ok=True)
    for backend, title in BACKENDS.items():
        entries = sorted(grouped[backend], key=lambda entry: entry[1]["env_id"])
        if not entries:
            continue
        lines += [
            f"## {title}\n",
            "Model observations: " + ", ".join(sorted({entry[1]["observation"] for entry in entries})) + ".\n",
            "| Environment | Model + planner | Quality | Model success | Random success | Model return | Random return | Episode / GT |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
        groups = {}
        for directory, result, config in entries:
            planned_name = result["planner_policy"]
            planned, random = result["evaluation"][planned_name], result["evaluation"]["random"]
            env_id = result["env_id"]
            name = f"[{env_id}]({link(notebooks[env_id])})" if env_id in notebooks else env_id
            gif = next((path for path in directory.glob("*.gif") if path.name != "random.gif"), None)
            visual = f"[GIF + GT]({link(gif)})" if gif else "—"
            quality = config.get("render_settings", {}).get("quality", "legacy")
            lines.append(
                f"| {name} | {planned_name} | {quality} | {planned['success_rate']:.0%} | {random['success_rate']:.0%} | "
                f"{planned['mean_return']:.3f} | {random['mean_return']:.3f} | {visual} |"
            )
            groups[env_id] = {
                policy: metrics.success_rate(torch.tensor(scores["successes"]).bool())
                for policy, scores in ((planned_name, planned), ("random", random))
            }
        figure = plots.bars(groups, xlabel="success rate (95% Wilson interval)", baseline="random")
        path = figures / f"{backend}_success.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        lines.append(f"\n![{title}: success rates]({link(path)})\n")
    lines += [
        "## Reproduce\n",
        "```bash",
        "python examples/envs/run_notebooks.py --output-dir runs/benchmarks/my-campaign",
        (
            "python examples/envs/report_results.py examples/envs/runs/benchmarks/my-campaign "
            "--output examples/envs/RESULTS.md"
        ),
        "```\n",
        (
            "Use `--physics-python` with the notebook runner when Isaac Lab/Newton live in a separate runtime. "
            "Every run records quality, effective resolutions, camera, seeds, model/planner settings, source hash, "
            "and artifact checksums. Earlier campaigns remain available; do not pool different render configurations."
        ),
    ]
    output.write_text("\n".join(lines) + "\n")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardware", help="Optional hardware description for this campaign")
    args = parser.parse_args()
    report_results(args.campaign, args.output, hardware=args.hardware)
