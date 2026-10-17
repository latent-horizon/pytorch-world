"""Observable-state staging with explicit, simulation-only scripted goal assistance.

The policy consumes fixed goal images and measured joint/gripper states. The
evaluation helper obtains those images from a separate successful scripted
rollout of each evaluation task. This is an assisted diagnostic, not zero-shot
final-goal planning, and cannot pass the physical progression gate.
"""

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from torch_world.planning import CEM
from torch_world.sim2real.config import PIECES, RandomizationConfig, fingerprint
from torch_world.sim2real.control import Observation
from torch_world.sim2real.evaluation import restore_run
from torch_world.sim2real.experiment import _folder, _json, save_video
from torch_world.sim2real.models import WorldModelPolicy
from torch_world.sim2real.reference import simulation_gate
from torch_world.sim2real.simulation import SO101Chess

PHASES = ("approach", "grasp", "lift", "transfer", "lower", "release", "clear", "withdraw", "return")


@dataclass
class GoalStage:
    name: str
    observation: Observation


class StagedWorldModelPolicy:
    """Plan toward supplied images; advance on measured arm/gripper proximity.

    Arm proximity is a stage transition signal, never evidence of grasp or
    placement success. Only the environment's unchanged success check scores
    a trial. No simulator state, piece location or expert action enters this
    policy. Intermediate goals must be supplied by the caller.
    """

    def __init__(
        self, adapter, stages, *, planner=None, seed=7, joint_tolerance=0.04, opening_tolerance=0.08, dwell_steps=2
    ):
        if not stages or joint_tolerance <= 0 or opening_tolerance <= 0 or dwell_steps < 1:
            raise ValueError("Need stages and positive proximity/dwell settings")
        shape = stages[0].observation.rgb.shape
        if any(s.observation.rgb.shape != shape for s in stages):
            raise ValueError("All stage images must have the same camera shape")
        if any(s.observation.contract != stages[0].observation.contract for s in stages):
            raise ValueError("All stage goals must share the same camera/calibration contract")
        self.stages = tuple(stages)
        self.joint_tolerance, self.opening_tolerance, self.dwell_steps = joint_tolerance, opening_tolerance, dwell_steps
        self.policy = WorldModelPolicy(adapter, stages[0].observation, planner=planner, seed=seed)
        self.adapter = adapter
        with torch.no_grad():
            self.targets = [
                adapter.encode(
                    s.observation.rgb[None, None].to(adapter.device),
                    s.observation.proprio[None, None].to(adapter.device),
                )[:, 0]
                for s in self.stages
            ]
        self.reset()

    def reset(self):
        self.policy.reset()
        self.index, self.dwell = 0, 0
        self.transitions = []
        self._select()

    def _select(self):
        self.policy.goal = self.stages[self.index].observation
        self.policy.goal_latent = self.targets[self.index]

    def __call__(self, observation):
        if self.policy.pending:
            raise RuntimeError("Record the applied action before the next observation")
        goal = self.stages[self.index].observation
        near = bool((observation.joints - goal.joints).abs().max() <= self.joint_tolerance)
        near = near and abs(observation.opening - goal.opening) <= self.opening_tolerance
        self.dwell = self.dwell + 1 if near else 0
        if self.dwell >= self.dwell_steps and self.index + 1 < len(self.stages):
            self.transitions.append({"stage": self.stages[self.index].name, "timestamp": observation.timestamp})
            self.index += 1
            self.dwell = 0
            self._select()
        return self.policy(observation)

    def record_action(self, action):
        self.policy.record_action(action)

    @property
    def goal(self):
        """Expose the common goal contract for the existing trial runner."""
        return self.stages[self.index].observation

    def check_latency(self, observation, control_dt, *, repeats=3):
        """Motor-free preflight with the same planner; restore initial stage."""
        self.reset()
        try:
            return self.policy.check_latency(observation, control_dt, repeats=repeats)
        finally:
            self.reset()


def scripted_goal_stages(env, *, seed):
    """Generate reference images in a dedicated simulation, retaining failures.

    Caller must reset/use a separate scene before learned execution. A failed
    scripted attempt provides no successful goal sequence and is not discarded
    from the evaluation denominator.
    """
    observation = env.reset(seed=seed)
    stages, failure = [], None
    source, target = env.source, env.target_square
    for _ in range(env.task.max_steps):
        phase = env.phase
        action = env.expert()
        try:
            observation = env.step(action)
        except ValueError as error:
            failure = str(error)
            break
        if env.phase != phase:
            stages.append(GoalStage(PHASES[phase], observation))
        if env.done:
            break
    successful = bool(env.info()["success"])
    if successful:
        stages.append(GoalStage("return", observation))
    metadata = {
        "success": successful,
        "source": source,
        "target": target,
        "steps": env.steps,
        "failure": failure,
        "randomization": env.episode_parameters,
    }
    return stages if successful else [], metadata


def evaluate_staged_run(
    training_run, *, randomization, output_dir, seeds, planner_factory=None, device="cpu", progress=None
):
    """Evaluate an unchanged checkpoint using scripted keyframe assistance."""
    training_run = Path(training_run).resolve()
    adapter, config, task, calibration = restore_run(training_run, device=device)
    data = torch.load(training_run / "episodes.pt", map_location="cpu", weights_only=True, mmap=True)
    used = {e["metadata"]["seed"] for group in data.values() for e in group}
    if not seeds or used.intersection(seeds):
        raise ValueError("Evaluation seeds must be nonempty and disjoint from training/validation")
    del data
    folder = _folder(output_dir)
    result = {
        "schema": 1,
        "model": adapter.recipe,
        "config": asdict(config),
        "task": asdict(task),
        "randomization": asdict(randomization),
        "training_run": str(training_run),
        "goal_protocol": "scripted_keyframes",
        "smoke": adapter.recipe["smoke"],
        "physical_results": None,
        "evaluation": {},
    }
    for domain, setting in (("nominal", RandomizationConfig(enabled=False)), ("randomized", randomization)):
        records = []
        for index, seed in enumerate(seeds):
            kwargs = {
                "piece": PIECES[index % 6],
                "color": ("white", "black")[(index // 6) % 2],
                "randomization": setting,
                "calibration": calibration,
            }
            with SO101Chess(config, task, **kwargs) as reference:
                stages, reference_info = scripted_goal_stages(reference, seed=seed)
            record = {
                "seed": seed,
                "piece": kwargs["piece"],
                "color": kwargs["color"],
                "source": reference_info["source"],
                "target": reference_info["target"],
                "reference": reference_info,
                "executed": bool(stages),
                "success": False,
                "lifted": False,
                "failure": "No successful scripted goal reference" if not stages else None,
                "failure_type": "goal_reference_unavailable" if not stages else None,
                "steps": 0,
                "mean_planning_seconds": 0.0,
                "max_planning_seconds": 0.0,
            }
            if stages:
                torch.save(
                    [
                        {
                            "name": s.name,
                            "rgb": s.observation.rgb,
                            "proprio": s.observation.proprio,
                            "timestamp": s.observation.timestamp,
                        }
                        for s in stages
                    ],
                    folder / f"{domain}-{seed}-goals.pt",
                )
                with SO101Chess(config, task, **kwargs) as env:
                    obs = env.reset(seed=seed)
                    policy = StagedWorldModelPolicy(
                        adapter, stages, seed=seed, planner=None if planner_factory is None else planner_factory()
                    )
                    frames, latencies = [], []
                    for _ in range(task.max_steps):
                        if index == 0:
                            frames.append((env.render_view(size=384, native=True) * 255).round().to(torch.uint8))
                        started = time.perf_counter()
                        action = policy(obs)
                        latencies.append(time.perf_counter() - started)
                        try:
                            obs = env.step(action)
                        except ValueError as error:
                            record.update(failure=str(error), failure_type="controller_rejection")
                            break
                        policy.record_action(env.last_command.applied)
                        if env.done:
                            break
                    record.update(
                        env.info(),
                        steps=env.steps,
                        stages_reached=policy.index,
                        final_stage=stages[policy.index].name,
                        stage_transitions=policy.transitions,
                        mean_planning_seconds=sum(latencies) / len(latencies),
                        max_planning_seconds=max(latencies),
                    )
                    if frames:
                        frames.append((env.render_view(size=384, native=True) * 255).round().to(torch.uint8))
                        save_video(folder / f"{domain}-staged.gif", frames, control_dt=config.control_dt)
            records.append(record)
            result["evaluation"][domain] = {
                "staged": {"episodes": records, "success_rate": sum(r["success"] for r in records) / len(records)}
            }
            _json(folder / "results.partial.json", result)
            event = {
                "stage": "staged_evaluation",
                "domain": domain,
                "episode": index + 1,
                "total": len(seeds),
                **record,
            }
            _json(folder / "progress.json", event)
            if progress:
                progress(event)
    _json(folder / "results.json", result)
    _json(folder / "manifest.json", {"complete": True, "schema": 1, "contract": fingerprint(config, task, calibration)})
    return {**result, "run_dir": str(folder)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_run", type=Path)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, help="Override source CEM sample count")
    parser.add_argument("--iterations", type=int, help="Override source CEM refinement count")
    parser.add_argument("--objective", choices=("final", "path"), default="final")
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = json.loads(args.source_report.read_text())
    execution = json.loads((args.source_report.parent / "execution.json").read_text())
    settings = dict(execution.get("planner", execution["arguments"]))
    if args.samples is not None:
        settings["samples"] = args.samples
    if args.iterations is not None:
        settings["iterations"] = args.iterations
    seeds = tuple(e["seed"] for e in source["evaluation"]["nominal"]["learned"]["episodes"])
    limits = torch.tensor([1, 1, 1, source["config"]["max_gripper_step"]], device=args.device)

    def planner():
        return CEM(
            settings["horizon"],
            4,
            num_samples=settings["samples"],
            num_elites=max(2, settings["samples"] // 8),
            iterations=settings["iterations"],
            bounds=(-limits, limits),
            objective=args.objective,
        )

    started = time.time()

    def progress(event):
        keys = ("stage", "domain", "episode", "total", "success", "lifted", "executed", "failure", "final_stage")
        print(
            json.dumps(
                {"elapsed_seconds": round(time.time() - started, 1), **{k: event[k] for k in keys if k in event}}
            ),
            flush=True,
        )

    result = evaluate_staged_run(
        args.training_run,
        randomization=RandomizationConfig(**source["randomization"]),
        output_dir=args.output,
        seeds=seeds,
        planner_factory=planner,
        device=args.device,
        progress=progress,
    )
    report = {
        "host": platform.node(),
        "arguments": vars(args),
        "elapsed_seconds": time.time() - started,
        "planner": {**{k: settings[k] for k in ("horizon", "samples", "iterations")}, "objective": args.objective},
        "gate": simulation_gate(result),
    }
    _json(Path(result["run_dir"]) / "execution.json", json.loads(json.dumps(report, default=str)))
    print(json.dumps({"run_dir": result["run_dir"], **report}, default=str), flush=True)


if __name__ == "__main__":
    main()
