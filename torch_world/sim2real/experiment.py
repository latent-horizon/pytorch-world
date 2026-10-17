"""Training, evaluation and portable run artifacts for the chess experiment."""

import json
import time
import uuid
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from torch_world.sim2real.config import ChessTaskConfig, RandomizationConfig, SO101Config, fingerprint
from torch_world.sim2real.data import EpisodeWindows, LeWAMWindows, cache_visual_features, collect_episodes
from torch_world.sim2real.models import WorldModelPolicy, make_adapter
from torch_world.sim2real.simulation import SO101Chess

SCHEMA = 1


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _folder(root):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    folder = Path(root) / f"{stamp}-{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def save_checkpoint(path, adapter, config, task, calibration=None):
    """Weights-only checkpoint, with an exact camera/action/task compatibility contract."""
    torch.save(
        {
            "schema": SCHEMA,
            "recipe": adapter.recipe,
            "contract": fingerprint(config, task, calibration),
            "config": asdict(config),
            "task": asdict(task),
            "calibration": None if calibration is None else asdict(calibration),
            "weights": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
        },
        path,
    )


def load_checkpoint(path, config, task, calibration=None, *, device="cpu", adapter=None):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("schema") != SCHEMA or saved.get("contract") != fingerprint(config, task, calibration):
        raise ValueError("Checkpoint camera/action/task/calibration contract does not match")
    recipe = saved["recipe"]
    if adapter is None:
        if recipe["encoder"] == "custom":
            raise ValueError("Supply the matching adapter when loading a custom encoder")
        adapter = make_adapter(
            recipe["kind"],
            size=recipe["size"],
            history=recipe["history"],
            smoke=recipe["smoke"],
            proprioception=recipe.get("proprioception", False),
            rollout_steps=recipe.get("rollout_steps", 1),
            chunk_size=recipe.get("chunk_size", 8),
            action_scale=recipe.get("action_scale", (1.0, 1.0, 1.0, 0.15)),
            command_state=recipe.get("command_state", False),
        )
    if adapter.recipe != recipe:
        raise ValueError("Model recipe does not match the checkpoint")
    adapter.load_state_dict(saved["weights"], strict=True)
    adapter.contract = saved["contract"]
    return adapter.to(device).eval()


def train(adapter, episodes, *, updates=1000, batch_size=32, seed=7, lr=3e-4, cache_features=False, progress=None):
    """Train only on supplied simulated episodes; no hardware collection path."""
    if updates < 0 or batch_size < 2:
        raise ValueError("updates must be nonnegative and batch_size at least two")
    features = (
        cache_visual_features(adapter, episodes, batch_size=batch_size, progress=progress) if cache_features else None
    )
    dataset = (
        LeWAMWindows(episodes, adapter.history, adapter.model.chunk_size)
        if adapter.kind == "lewam"
        else EpisodeWindows(episodes, adapter.training_window, features)
    )
    if len(dataset) < 2:
        raise ValueError("Need at least two windows for training normalization")
    if adapter.kind == "lewam" and not any(
        episodes[e].metadata.get("outcome", {}).get("success", False) for e, _ in dataset.index
    ):
        warnings.warn("No successful LeWAM demonstrations: the action branch receives no supervision.", stacklevel=2)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr)
    adapter.train()
    losses = []
    for _ in range(updates):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {k: v.to(adapter.device) for k, v in batch.items()}
        components = (
            adapter.loss_features(batch["features"], batch["actions"], batch["states"])
            if features is not None
            else adapter.loss(**batch)
        )
        loss = components["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite world-model training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        if progress is not None and (len(losses) == 1 or len(losses) % 100 == 0 or len(losses) == updates):
            progress(
                {
                    "stage": "training",
                    "update": len(losses),
                    "total": updates,
                    **{key: float(value.detach()) for key, value in components.items()},
                }
            )
    adapter.eval()
    return losses


@torch.no_grad()
def heldout_loss(adapter, episodes):
    dataset = (
        LeWAMWindows(episodes, adapter.history, adapter.model.chunk_size)
        if adapter.kind == "lewam"
        else EpisodeWindows(episodes, adapter.training_window)
    )
    loader = DataLoader(dataset, batch_size=32)
    total, count = 0.0, 0
    for batch in loader:
        batch = {k: v.to(adapter.device) for k, v in batch.items()}
        loss = adapter.loss(**batch)["loss"]
        total += float(loss) * len(batch["pixels"])
        count += len(batch["pixels"])
    return total / count


def evaluate(
    adapter,
    config,
    task,
    randomization,
    *,
    seeds=(10007, 10008),
    steps=None,
    calibration=None,
    planner_factory=None,
    render_size=384,
    progress=None,
    policies=("learned", "scripted", "random"),
):
    """Compare on identical seeds and goals. Controller rejection is a recorded failure.

    Select ``policies`` for independent evaluations in short processes. Keep the
    full ordered seed list: its indices determine piece and color assignments.
    """
    from torch_world.sim2real.config import PIECES

    if not policies or len(set(policies)) != len(policies) or set(policies) - {"learned", "scripted", "random"}:
        raise ValueError("Choose distinct learned, scripted or random policies")
    result, videos = {}, {}
    for name in policies:
        outcomes = []
        for index, seed in enumerate(seeds):
            with SO101Chess(
                config,
                task,
                piece=PIECES[index % 6],
                color=("white", "black")[(index // 6) % 2],
                randomization=randomization,
                calibration=calibration,
            ) as env:
                obs = env.reset(seed=seed)
                goal = env.goal()
                policy = (
                    WorldModelPolicy(
                        adapter, goal, seed=seed, planner=None if planner_factory is None else planner_factory()
                    )
                    if name == "learned"
                    else None
                )
                rng = torch.Generator().manual_seed(seed)
                frames = (
                    [(env.render_view(size=render_size, native=True) * 255).round().to(torch.uint8)]
                    if index == 0
                    else None
                )
                failure = None
                latencies = []
                for _ in range(min(steps or task.max_steps, task.max_steps)):
                    started = time.perf_counter()
                    action = (
                        policy(obs)
                        if policy
                        else env.expert()
                        if name == "scripted"
                        else torch.rand(4, generator=rng) * 2 - 1
                    )
                    latencies.append(time.perf_counter() - started)
                    try:
                        obs = env.step(action)
                    except ValueError as error:
                        failure = str(error)
                        break
                    if policy:
                        policy.record_action(env.last_command.applied)
                    if frames is not None:
                        frames.append((env.render_view(size=render_size, native=True) * 255).round().to(torch.uint8))
                    if env.done:
                        break
                outcomes.append(
                    {
                        **env.info(),
                        "seed": seed,
                        "piece": env.piece,
                        "color": env.color,
                        "source": env.source,
                        "target": env.target_square,
                        "steps": env.steps,
                        "failure": failure,
                        "randomization": env.episode_parameters,
                        "mean_planning_seconds": sum(latencies) / len(latencies),
                        "max_planning_seconds": max(latencies),
                    }
                )
                if progress is not None:
                    progress(
                        {
                            "stage": "evaluation",
                            "policy": name,
                            "episode": index + 1,
                            "total": len(seeds),
                            **outcomes[-1],
                        }
                    )
                if frames is not None:
                    videos[name] = torch.stack(frames)
        result[name] = {"success_rate": sum(o["success"] for o in outcomes) / len(outcomes), "episodes": outcomes}
    return result, videos


def save_video(path, frames, *, control_dt=0.1):
    """Save actual RGB frames as an animated GIF; requires the notebooks extra."""
    from PIL import Image

    images = [
        Image.fromarray(
            (frame if frame.dtype == torch.uint8 else (frame.clamp(0, 1) * 255).round().to(torch.uint8))
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        for frame in frames
    ]
    images[0].save(path, save_all=True, append_images=images[1:], loop=0, duration=round(control_dt * 1000))


def run_experiment(
    *,
    config=None,
    task=None,
    randomization=None,
    calibration=None,
    model="lewm",
    adapter=None,
    smoke=False,
    episodes=24,
    updates=1000,
    collect_steps=None,
    evaluation_steps=None,
    evaluation_seeds=(10007, 10008),
    device="cpu",
    output_dir=None,
    save_gifs=True,
    planner_factory=None,
    batch_size=32,
    cache_features=False,
    progress=None,
):
    """Simulation-only training and nominal/randomized evaluation.

    Defaults are an initial experimental budget, not a claim of zero-shot
    transfer. Use twelve or more evaluation seeds to cover both colors/types.
    Real calibration must be supplied before training a hardware-bound model.
    """
    config, task = config or SO101Config(), task or ChessTaskConfig()
    randomization = randomization or RandomizationConfig()
    folder = _folder(output_dir) if output_dir is not None else None

    def report(event):
        if folder is not None:
            _json(folder / "progress.json", event)
        if progress is not None:
            progress(event)

    if episodes < 1 or not evaluation_seeds or (evaluation_steps is not None and evaluation_steps < 1):
        raise ValueError("Need training episodes and positive evaluation budget")
    train_seeds = set(range(config.seed, config.seed + episodes))
    holdout_seed = config.seed + episodes + 1000
    holdout_count = max(1, episodes // 4)
    if train_seeds & set(evaluation_seeds) or set(range(holdout_seed, holdout_seed + holdout_count)) & set(
        evaluation_seeds
    ):
        raise ValueError("Training, holdout and evaluation seeds must be disjoint")
    torch.manual_seed(config.seed)
    adapter = (
        adapter or make_adapter(model, size=config.size, smoke=smoke, action_scale=(1, 1, 1, config.max_gripper_step))
    ).to(device)
    if adapter.recipe["size"] != config.size:
        raise ValueError("Model image size and controller configuration differ")
    train_data = collect_episodes(
        config,
        task,
        randomization,
        count=episodes,
        seed=config.seed,
        steps=collect_steps,
        calibration=calibration,
        progress=report,
        include_commanded=adapter.recipe.get("command_state", False),
    )
    test_data = collect_episodes(
        config,
        task,
        randomization,
        count=holdout_count,
        seed=holdout_seed,
        steps=collect_steps,
        calibration=calibration,
        progress=report,
        include_commanded=adapter.recipe.get("command_state", False),
    )
    if folder is not None:
        torch.save(
            {"train": [asdict(e) for e in train_data], "holdout": [asdict(e) for e in test_data]},
            folder / "episodes.pt",
        )
    losses = train(
        adapter,
        train_data,
        updates=updates,
        seed=config.seed,
        batch_size=batch_size,
        cache_features=cache_features,
        progress=report,
    )
    adapter.contract = fingerprint(config, task, calibration)
    if folder is not None:
        save_checkpoint(folder / "model.pt", adapter, config, task, calibration)
        _json(folder / "losses.json", losses)
    validation = heldout_loss(adapter, test_data)
    evaluations, videos = {}, {}
    for name, setting in (("nominal", RandomizationConfig(enabled=False)), ("randomized", randomization)):
        evaluations[name], videos[name] = evaluate(
            adapter,
            config,
            task,
            setting,
            seeds=evaluation_seeds,
            steps=evaluation_steps,
            calibration=calibration,
            planner_factory=planner_factory,
            progress=lambda event, domain=name: report({**event, "domain": domain}),
        )
    result = {
        "schema": SCHEMA,
        "model": adapter.recipe,
        "losses": losses,
        "heldout_loss": validation,
        "evaluation": evaluations,
        "config": asdict(config),
        "task": asdict(task),
        "randomization": asdict(randomization),
        "calibration_id": None if calibration is None else calibration.id,
        "training_episodes": [e.metadata for e in train_data],
        "holdout_episodes": [e.metadata for e in test_data],
        "updates": updates,
        "smoke": smoke,
        "physical_results": None,
    }
    if folder is not None:
        _json(folder / "results.json", result)
        torch.save(videos, folder / "videos.pt")
        if save_gifs:
            for domain, policies in videos.items():
                for name, frames in policies.items():
                    save_video(folder / f"{domain}-{name}.gif", frames, control_dt=config.control_dt)
        _json(
            folder / "manifest.json",
            {"schema": SCHEMA, "complete": True, "contract": fingerprint(config, task, calibration)},
        )
        result["run_dir"] = str(folder)
    return {**result, "adapter": adapter, "videos": videos}


def run_trial(robot, policy, *, steps=120, output_dir, save_gif=True):
    """Record an explicitly connected robot trial, including interrupted trials.

    Success remains unknown until ``label_trial`` is called by the operator.
    A latency preflight runs before any policy action is dispatched.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    if robot.hardware:
        if getattr(policy.adapter, "contract", None) != robot.contract or policy.goal.contract != robot.contract:
            raise ValueError("Model and goal must match this hardware calibration/control contract")
        if policy.adapter.recipe["smoke"]:
            raise ValueError("Smoke-test models are not hardware experiments")
    folder = _folder(output_dir)
    frames, log_start = [], len(robot.log)
    failure = None
    try:
        obs = robot.reset()
        policy.check_latency(obs, robot.config.control_dt)
        policy.reset()
        for _ in range(steps):
            obs = robot.observe()
            frames.append(obs.rgb)
            started = time.perf_counter()
            action = policy(obs)
            if time.perf_counter() - started > robot.config.control_dt:
                raise TimeoutError("Planner exceeded the configured control interval")
            obs = robot.step(action, observation=obs)
            policy.record_action(robot.last_command.applied)
        frames.append(obs.rgb)
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        robot.stop()
        _json(
            folder / "trial.json",
            {
                "schema": SCHEMA,
                "contract": robot.contract,
                "success": None,
                "hardware": robot.hardware,
                "dry_run": robot.dry_run,
                "failure": failure,
                "config": asdict(robot.config),
                "task": asdict(robot.task),
                "calibration": asdict(robot.calibration),
                "model": policy.adapter.recipe,
                "commands": robot.log[log_start:],
            },
        )
        if frames:
            torch.save(torch.stack(frames), folder / "frames.pt")
            if save_gif:
                save_video(folder / "trial.gif", frames, control_dt=robot.config.control_dt)
    return folder


def label_trial(folder, success, *, notes=""):
    """Attach an operator-observed outcome to a completed real trial."""
    path = Path(folder) / "trial.json"
    result = json.loads(path.read_text())
    if not result["hardware"] or result["dry_run"]:
        raise ValueError("A mock/dry run cannot be labeled as a physical success")
    if not isinstance(success, bool):
        raise TypeError("success must be a boolean operator observation")
    result.update(success=success, operator_notes=notes, label_source="operator")
    _json(path, result)


def scripted_episode(
    *,
    config=None,
    task=None,
    piece="pawn",
    color="white",
    seed=7,
    source="c4",
    target="d4",
    calibration=None,
    render_size=512,
):
    """Record a nominal scripted reference, independently of learned-model results."""
    with SO101Chess(config, task, piece=piece, color=color, calibration=calibration) as env:
        obs = env.reset(seed=seed, source=source, target=target)
        start, goal = obs, env.goal()
        views = {
            "start": env.render_view(size=render_size, native=True),
            "goal": env.goal(size=render_size, native=True).rgb,
        }
        frames = [(env.world.render(camera="front", size=render_size)[0] * 255).round().to(torch.uint8)]
        for _ in range(env.task.max_steps):
            env.step(env.expert())
            frames.append((env.world.render(camera="front", size=render_size)[0] * 255).round().to(torch.uint8))
            if env.done:
                break
        views["final"] = env.render_view(size=render_size, native=True)
        return {
            "start": start,
            "goal": goal,
            "final": env.observe(),
            "frames": torch.stack(frames),
            "views": views,
            "outcome": env.info(),
            "source": source,
            "target": target,
            "piece": piece,
            "color": color,
        }
