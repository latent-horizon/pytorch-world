"""Simulation-only expert takeovers and privileged-state LeWAM diagnostics.

Takeovers are one round of trajectory aggregation, not online DAgger labels.
Oracle state is never attached to ordinary/hardware observations or checkpoints.
"""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from torch_world.planning import LeWAMPlanner
from torch_world.sim2real.config import RandomizationConfig
from torch_world.sim2real.curriculum import TASK, recovery_schedule, write_json
from torch_world.sim2real.data import Episode
from torch_world.sim2real.evaluation import restore_run
from torch_world.sim2real.experiment import save_video, train
from torch_world.sim2real.investigation import action_tracking, trace_episode
from torch_world.sim2real.models import WorldModelPolicy
from torch_world.sim2real.privileged import OraclePolicy, make_oracle, oracle_observation
from torch_world.sim2real.reference import reference_setup
from torch_world.sim2real.simulation import SO101Chess

SEEDS = (17, 29, 43)
VARIANTS = ("image_control", "image_corrections", "oracle_control", "oracle_corrections")


class ExpertTakeover:
    """Replay an actual learned prefix, then align the expert to physical state.

    Nearest-reference phase is used only by the collection expert. It is never
    provided to either learned policy. Failed rescue attempts remain failures.
    """

    def __init__(self, env, prefix, reference):
        self.env, self.prefix, self.reference = env, prefix, reference
        self.index, self.anchor = 0, None
        self.latencies = []

    def __call__(self, observation):
        if self.index < len(self.prefix):
            action = self.prefix[self.index]
        else:
            if self.anchor is None:
                env = self.env
                distance = [
                    np.linalg.norm((np.array(r["tool"]) - env.tool_position) / 0.01) ** 2
                    + np.linalg.norm((np.array(r["piece"]) - env.position) / 0.01) ** 2
                    + ((r["opening"] - env.opening) / 0.15) ** 2
                    for r in self.reference
                ]
                self.anchor = int(np.argmin(distance))
                env.phase = self.reference[self.anchor]["phase"]
                env.phase_steps = env._grip_stable = 0
                env._previous_opening = env.opening
                if env.phase >= 2:
                    env.carry_offset = env.position - env.tool_position
            action = self.env.expert()
        self.index += 1
        return action

    def record_action(self, action):
        pass


def _trace(env, **kwargs):
    states = []
    episode, report = trace_episode(
        env,
        include_commanded=True,
        capture=lambda e: states.append(oracle_observation(e, e.observe()).privileged),
        **kwargs,
    )
    episode.proprio = torch.stack(states)
    return episode, report


def evaluation_cases():
    return [{"name": "nominal", "seed": 90000, "perturbations": {}}] + [
        {"name": f"disturbance_{step}", "seed": 90001 + i, "perturbations": recovery_schedule(90001 + i, step)}
        for i, step in enumerate((4, 16, 50, 73))
    ]


def prepare(previous, output):
    """Freeze collection and evaluation protocol before any correction training."""
    from dataclasses import asdict

    previous, output = Path(previous).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config, task = reference_setup(size=64)
    report = {
        "model_seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "updates": 2000,
        "batch_size": 32,
        "evaluation_cases": evaluation_cases(),
        "policy_seeds": [101, 202, 303],
        "noise_scales": [1.0, 0.0],
        "previous": str(previous),
        "training_sources": {},
        "scope": "same pawn move, one round of takeovers; not new-task validation",
        "eligible_for_sim2real_gate": False,
        "physical_results": None,
        "original_replays": [],
        "collection": [],
        "corrections": [],
        "controls": [],
    }
    for seed in SEEDS:
        checkpoint = previous / "models" / f"h16_recovery_command-seed{seed}" / "model.pt"
        report["training_sources"][str(seed)] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    write_json(output / "protocol.json", report)
    originals = torch.load(previous / "prepared/episodes.pt", weights_only=True)["recovery"]
    clean, corrections = [], []
    with SO101Chess(config, task, randomization=RandomizationConfig(enabled=False)) as env:
        for original in originals:
            metadata = original["metadata"]
            ep, result = _trace(
                env,
                seed=metadata["seed"],
                source=metadata["source"],
                target=metadata["target"],
                replay=original["requested_actions"],
            )
            torch.testing.assert_close(ep.actions, original["actions"], rtol=0, atol=1e-6)
            torch.testing.assert_close(ep.proprio[:, :10], original["proprio"], rtol=0, atol=1e-6)
            ep.action_valid = original["action_valid"]
            clean.append(ep)
            report["original_replays"].append(result)
        reference = report["original_replays"][0]["trace"]
        # Replays omit expert phases; obtain the identical scripted reference.
        _, scripted = _trace(env, **TASK)
        reference = scripted["trace"]
        for seed in SEEDS:
            adapter, _, _, _ = restore_run(previous / "models" / f"h16_recovery_command-seed{seed}")
            for repeat in range(4):
                collection_seed = 61000 + seed * 10 + repeat

                def factory(goal, adapter=adapter, collection_seed=collection_seed):
                    return WorldModelPolicy(adapter, goal, seed=collection_seed, planner=LeWAMPlanner())

                _, failure = _trace(env, **{**TASK, "seed": collection_seed}, policy_factory=factory)
                failure.update(model_seed=seed, policy_seed=collection_seed)
                report["collection"].append(failure)
                if failure["outcome"]["success"]:
                    continue
                applied_rows = [r for r in failure["trace"] if "applied" in r]
                cuts = sorted(
                    {
                        max(1, int(len(applied_rows) * fraction))
                        for fraction in (0.25, 0.5, 0.75)
                        if len(applied_rows) >= 2
                    }
                )
                for cut in cuts:
                    prefix = [r["requested"] for r in applied_rows[:cut]]
                    takeover = ExpertTakeover(env, prefix, reference)
                    ep, result = _trace(
                        env,
                        **{**TASK, "seed": collection_seed},
                        policy_factory=lambda goal, takeover=takeover: takeover,
                    )
                    ep.action_valid[:cut] = 0
                    ep.metadata.update(correction=True, prefix_steps=cut, source_model_seed=seed)
                    result.update(prefix_steps=cut, source_model_seed=seed, reference_anchor=takeover.anchor)
                    corrections.append(ep)
                    report["corrections"].append(result)
                    write_json(output / "collection-progress.json", report)
                    print("correction", seed, repeat, cut, result["outcome"]["success"], result["failure"], flush=True)
        for case in report["evaluation_cases"]:
            _, control = _trace(env, **{**TASK, "seed": case["seed"]}, perturbations=case["perturbations"])
            control["case"] = case["name"]
            report["controls"].append(control)
    torch.save(
        {"original": [asdict(e) for e in clean], "corrected": [asdict(e) for e in clean + corrections]},
        output / "episodes.pt",
    )
    report["dataset_sha256"] = hashlib.sha256((output / "episodes.pt").read_bytes()).hexdigest()
    write_json(output / "protocol.json", report)
    return report


def pretrain_oracle(prepared, output, seed, *, device="cpu"):
    """Match the image model's 2,000-update original-data pretraining stage."""
    prepared, output = Path(prepared), Path(output)
    protocol = json.loads((prepared / "protocol.json").read_text())
    if seed not in SEEDS:
        raise ValueError("Unknown model seed")
    if hashlib.sha256((prepared / "episodes.pt").read_bytes()).hexdigest() != protocol["dataset_sha256"]:
        raise ValueError("Dataset changed")
    output.mkdir(parents=True, exist_ok=False)
    episodes = [Episode(**e) for e in torch.load(prepared / "episodes.pt", weights_only=True)["original"]]
    adapter = make_oracle(seed).to(device)
    torch.manual_seed(seed + 100000)
    losses = train(
        adapter,
        episodes,
        updates=protocol["updates"],
        batch_size=protocol["batch_size"],
        seed=seed,
        progress=lambda event: write_json(output / "progress.json", event),
    )
    torch.save(
        {
            "simulation_diagnostic": True,
            "oracle": True,
            "seed": seed,
            "weights": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
        },
        output / "model.pt",
    )
    write_json(
        output / "training.json",
        {
            "seed": seed,
            "updates": protocol["updates"],
            "losses": losses,
            "dataset_sha256": protocol["dataset_sha256"],
            "device": device,
        },
    )


def train_variant(prepared, output, variant, seed, *, device="cpu", updates=None, source_root=None):
    prepared, output = Path(prepared), Path(output)
    protocol = json.loads((prepared / "protocol.json").read_text())
    if variant not in VARIANTS or seed not in SEEDS:
        raise ValueError("Unknown variant/seed")
    if hashlib.sha256((prepared / "episodes.pt").read_bytes()).hexdigest() != protocol["dataset_sha256"]:
        raise ValueError("Dataset changed")
    output.mkdir(parents=True, exist_ok=False)
    oracle = variant.startswith("oracle")
    dataset = "corrected" if variant.endswith("corrections") else "original"
    episodes = [Episode(**e) for e in torch.load(prepared / "episodes.pt", weights_only=True)[dataset]]
    if oracle:
        adapter = make_oracle(seed)
        initial = prepared.parent / "oracle_initial" / f"seed{seed}"
        baseline = json.loads((initial / "training.json").read_text())
        if baseline["updates"] != protocol["updates"] or baseline["dataset_sha256"] != protocol["dataset_sha256"]:
            raise ValueError("Oracle pretraining budget or dataset differs")
        adapter.load_state_dict(torch.load(initial / "model.pt", weights_only=True)["weights"])
    else:
        source = Path(source_root or protocol["previous"]) / "models" / f"h16_recovery_command-seed{seed}"
        if hashlib.sha256((source / "model.pt").read_bytes()).hexdigest() != protocol["training_sources"][str(seed)]:
            raise ValueError("Source checkpoint changed")
        adapter, _, _, _ = restore_run(source)
        episodes = [replace(e, proprio=e.proprio[:, :10]) for e in episodes]
    adapter.to(device)
    torch.manual_seed(seed + 200000)
    budget = protocol["updates"] if updates is None else updates
    losses = train(
        adapter,
        episodes,
        updates=budget,
        batch_size=protocol["batch_size"],
        seed=seed,
        progress=lambda event: write_json(output / "progress.json", event),
    )
    torch.save(
        {
            "simulation_diagnostic": True,
            "oracle": oracle,
            "seed": seed,
            "variant": variant,
            "weights": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
        },
        output / "model.pt",
    )
    result = {
        "variant": variant,
        "seed": seed,
        "updates": budget,
        "losses": losses,
        "dataset": dataset,
        "dataset_sha256": protocol["dataset_sha256"],
        "episodes": len(episodes),
        "successful_episodes": sum(e.metadata["outcome"]["success"] for e in episodes),
        "initialization": "2000-update original-data oracle checkpoint"
        if oracle
        else "existing 2000-update image checkpoint",
        "simulation_only": True,
        "device": device,
    }
    write_json(output / "training.json", result)
    return result


def evaluate_variant(prepared, trained):
    prepared, trained = Path(prepared), Path(trained)
    if (trained / "evaluation.json").exists():
        raise FileExistsError("Preserve completed evaluations")
    protocol = json.loads((prepared / "protocol.json").read_text())
    training = json.loads((trained / "training.json").read_text())
    checkpoint = torch.load(trained / "model.pt", weights_only=True)
    if training["dataset_sha256"] != protocol["dataset_sha256"]:
        raise ValueError("Mismatched training dataset")
    if checkpoint["oracle"]:
        adapter = make_oracle(training["seed"])
    else:
        source = Path(protocol["previous"]) / "models" / f"h16_recovery_command-seed{training['seed']}"
        adapter, _, _, _ = restore_run(source)
    adapter.load_state_dict(checkpoint["weights"])
    adapter.eval()
    config, task = reference_setup(size=64)
    report = {
        "variant": training["variant"],
        "seed": training["seed"],
        "trials": [],
        "eligible_for_sim2real_gate": False,
    }
    with SO101Chess(config, task, randomization=RandomizationConfig(enabled=False)) as env:
        for noise in protocol["noise_scales"]:
            for case in protocol["evaluation_cases"]:
                for policy_seed in protocol["policy_seeds"] if noise and case["name"] == "nominal" else [101]:

                    def factory(goal, policy_seed=policy_seed, noise=noise):
                        if checkpoint["oracle"]:
                            return OraclePolicy(adapter, env, goal, seed=policy_seed, noise=noise)
                        return WorldModelPolicy(
                            adapter, goal, seed=policy_seed, planner=LeWAMPlanner(noise_scale=noise)
                        )

                    ep, result = trace_episode(
                        env,
                        **{**TASK, "seed": case["seed"]},
                        policy_factory=factory,
                        perturbations={int(k): v for k, v in case["perturbations"].items()},
                    )
                    result.update(case=case["name"], policy_seed=policy_seed, noise_scale=noise)
                    report["trials"].append(result)
                    write_json(trained / "evaluation-progress.json", report)
                    if case["name"] == "nominal" and policy_seed == 101:
                        save_video(trained / f"nominal-{noise}.gif", ep.pixels, control_dt=config.control_dt)
                    print(
                        training["variant"],
                        training["seed"],
                        noise,
                        case["name"],
                        result["outcome"]["success"],
                        flush=True,
                    )
    write_json(trained / "evaluation.json", report)
    return report


def teacher_forced_tracking(prepared, trained):
    """Compare action fit on a common clean demonstration, separately from control."""
    prepared, trained = Path(prepared), Path(trained)
    protocol = json.loads((prepared / "protocol.json").read_text())
    checkpoint = torch.load(trained / "model.pt", weights_only=True)
    training = json.loads((trained / "training.json").read_text())
    if not checkpoint.get("simulation_diagnostic") or any(checkpoint[k] != training[k] for k in ("seed", "variant")):
        raise ValueError("Mismatched diagnostic checkpoint")
    if checkpoint["oracle"]:
        adapter = make_oracle(training["seed"])
    else:
        source = Path(protocol["previous"]) / "models" / f"h16_recovery_command-seed{training['seed']}"
        adapter, _, _, _ = restore_run(source)
    adapter.load_state_dict(checkpoint["weights"])
    adapter.eval()
    episode = Episode(**torch.load(prepared / "episodes.pt", weights_only=True)["original"][0])
    if not checkpoint["oracle"]:
        episode = replace(episode, proprio=episode.proprio[:, :10])
    tracking = {
        str(noise): action_tracking(adapter, episode, planner=LeWAMPlanner(noise_scale=noise))
        for noise in protocol["noise_scales"]
    }
    write_json(trained / "tracking.json", tracking)
    return tracking


def summarize(prepared, models, output):
    """Require a complete matched matrix and retain seed-level evidence."""
    prepared, models = Path(prepared), Path(models)
    protocol = json.loads((prepared / "protocol.json").read_text())

    def compact(value):
        if isinstance(value, dict):
            return {k: compact(v) for k, v in value.items() if k != "trace"}
        if isinstance(value, list):
            return [compact(v) for v in value]
        return value

    report = compact(protocol)
    report["runs"], report["summary"], report["oracle_pretraining"] = [], [], []
    expected = {
        (noise, case["name"], policy_seed)
        for noise in protocol["noise_scales"]
        for case in protocol["evaluation_cases"]
        for policy_seed in (protocol["policy_seeds"] if noise and case["name"] == "nominal" else [101])
    }
    for seed in protocol["model_seeds"]:
        initial = models.parent / "oracle_initial" / f"seed{seed}"
        metadata = json.loads((initial / "training.json").read_text())
        if metadata["dataset_sha256"] != protocol["dataset_sha256"] or metadata["updates"] != protocol["updates"]:
            raise ValueError("Oracle pretraining is not matched")
        report["oracle_pretraining"].append(metadata)
        for variant in protocol["variants"]:
            folder = models / f"{variant}-seed{seed}"
            training = json.loads((folder / "training.json").read_text())
            evaluation = json.loads((folder / "evaluation.json").read_text())
            if any(r["variant"] != variant or r["seed"] != seed for r in (training, evaluation)):
                raise ValueError("Mismatched run identity")
            if training["dataset_sha256"] != protocol["dataset_sha256"] or training["updates"] != protocol["updates"]:
                raise ValueError("Mismatched training data or budget")
            trials = evaluation["trials"]
            actual = [(t["noise_scale"], t["case"], t["policy_seed"]) for t in trials]
            if len(actual) != len(expected) or set(actual) != expected:
                raise ValueError("Incomplete or duplicated evaluation trials")
            report["runs"].append(
                {
                    "training": training,
                    "trials": compact(trials),
                    "tracking": json.loads((folder / "tracking.json").read_text())
                    if (folder / "tracking.json").exists()
                    else None,
                    "checkpoint_sha256": hashlib.sha256((folder / "model.pt").read_bytes()).hexdigest(),
                }
            )
    for variant in protocol["variants"]:
        for noise in protocol["noise_scales"]:
            for domain in ("nominal", "disturbed"):
                by_seed = {
                    str(run["training"]["seed"]): [
                        t
                        for t in run["trials"]
                        if t["noise_scale"] == noise and (t["case"] == "nominal") == (domain == "nominal")
                    ]
                    for run in report["runs"]
                    if run["training"]["variant"] == variant
                }
                trials = [t for group in by_seed.values() for t in group]
                report["summary"].append(
                    {
                        "variant": variant,
                        "noise_scale": noise,
                        "domain": domain,
                        "trials": len(trials),
                        "successes": sum(t["outcome"]["success"] for t in trials),
                        "lifts": sum(t["outcome"]["lifted"] for t in trials),
                        "rejections": sum(t["failure"] is not None for t in trials),
                        "success_by_model_seed": {
                            seed: sum(t["outcome"]["success"] for t in group) for seed, group in by_seed.items()
                        },
                    }
                )
    report["complete"] = True
    report["limitations"] = [
        "One training move and one correction round; fresh disturbance draws do not establish task generalization.",
        "Oracle uses exact 28-coordinate physical state, a constant visual branch and no SIGReg weight; it is a diagnostic, not a pure camera-only ablation.",
        "No phase/counter labels reach either learned policy. Expert handoff phase uses nearest scripted physical state only during collection.",
        "Only successful expert tails receive imitation supervision; failed rescue transitions still supervise dynamics.",
        "Both modalities receive 2000 original-data pretraining and 2000 continuation updates; within each modality the two continuations share a checkpoint.",
        "Repeated policy seeds share models and scenes; no independent-trial confidence intervals are claimed.",
        "Successful privileged-state control would implicate the observation/representation pathway; failure does not rule out a perception problem.",
    ]
    write_json(output, report)
    return report


def run_study(previous, output, *, device="cpu"):
    """Run the simulation-only protocol in a fresh directory."""
    output = Path(output)
    prepare(previous, output / "prepared")
    for seed in SEEDS:
        pretrain_oracle(output / "prepared", output / "oracle_initial" / f"seed{seed}", seed, device=device)
        for variant in VARIANTS:
            folder = output / "models" / f"{variant}-seed{seed}"
            train_variant(output / "prepared", folder, variant, seed, device=device)
            evaluate_variant(output / "prepared", folder)
            teacher_forced_tracking(output / "prepared", folder)
    return summarize(output / "prepared", output / "models", output / "results.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("previous", type=Path)
    p.add_argument("output", type=Path)
    p = sub.add_parser("pretrain-oracle")
    p.add_argument("prepared", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("train")
    p.add_argument("prepared", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--source-root", type=Path)
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("evaluate")
    p.add_argument("prepared", type=Path)
    p.add_argument("trained", type=Path)
    p = sub.add_parser("summarize")
    p.add_argument("prepared", type=Path)
    p.add_argument("models", type=Path)
    p.add_argument("output", type=Path)
    p = sub.add_parser("run")
    p.add_argument("previous", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "prepare":
        prepare(args.previous, args.output)
    elif args.command == "pretrain-oracle":
        pretrain_oracle(args.prepared, args.output, args.seed, device=args.device)
    elif args.command == "train":
        train_variant(
            args.prepared, args.output, args.variant, args.seed, device=args.device, source_root=args.source_root
        )
    elif args.command == "summarize":
        summarize(args.prepared, args.models, args.output)
    elif args.command == "run":
        run_study(args.previous, args.output, device=args.device)
    else:
        evaluate_variant(args.prepared, args.trained)


if __name__ == "__main__":
    main()
