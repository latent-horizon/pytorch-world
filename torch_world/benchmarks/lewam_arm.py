"""LeWAM diagnosis on our registered ObstacleArm-v0 environment.

This is an owned-environment adaptation, not the paper's DMControl benchmark.
All learned policies are evaluated on identical, previously unseen task seeds.
"""

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from torch_world.envs import make
from torch_world.models import LeWAM


@dataclass(frozen=True)
class ArmConfig:
    seed: int = 17
    train_episodes: int = 128
    validation_episodes: int = 16
    test_episodes: int = 24
    steps: int = 100
    size: int = 64
    history: int = 4
    chunk_size: int = 5
    updates: int = 4000
    batch_size: int = 32
    flow_steps: int = 16
    mean_samples: int = 16
    validation_interval: int = 200
    reference: bool = False
    frameskip: int = 1
    history_stride: int = 1
    launch_fraction: float = 0.0
    phase_validation: bool = False
    execution_interval: int | None = None
    expert_gain: float = 10.0
    recovery_fraction: float = 0.0
    goal_mode: str = "terminal"
    goal_horizon: int = 10
    goal_marker: bool = False
    diverse_motion: bool = False
    test_seed_offset: int = 20000

    def __post_init__(self):
        budgets = {
            key: value
            for key, value in asdict(self).items()
            if key
            not in (
                "seed",
                "reference",
                "launch_fraction",
                "phase_validation",
                "execution_interval",
                "expert_gain",
                "recovery_fraction",
                "goal_mode",
                "goal_marker",
                "diverse_motion",
            )
        }
        if self.goal_mode not in ("terminal", "sampled"):
            raise ValueError("goal_mode must be terminal or sampled")
        if self.goal_mode == "sampled" and (not self.reference or self.history_stride != self.frameskip):
            raise ValueError("Sampled-goal planning requires reference and history_stride=frameskip")
        if not 0 <= self.recovery_fraction <= 1:
            raise ValueError("recovery_fraction must be in [0,1]")
        if not np.isfinite(self.expert_gain) or self.expert_gain <= 0:
            raise ValueError("expert_gain must be finite and positive")
        if self.execution_interval is not None and not 1 <= self.execution_interval <= self.chunk_size:
            raise ValueError("execution_interval must lie between one and chunk_size")
        if not 0 <= self.launch_fraction <= 1:
            raise ValueError("launch_fraction must be in [0,1]")
        if not self.reference and (self.launch_fraction or self.phase_validation):
            raise ValueError("Launch sampling and phase validation require reference=True")
        if self.seed < 0 or min(budgets.values()) < 1 or self.chunk_size >= self.steps:
            raise ValueError("Positive budgets and complete action windows are required")
        if self.reference and (self.size < 202 or self.frameskip > self.chunk_size):
            raise ValueError("Reference encoder requires size >=202 and frameskip <= chunk_size")
        if not self.reference and (self.frameskip != 1 or self.history_stride != 1):
            raise ValueError("Spaced history/dynamics require reference=True")
        if self.reference and self.history_stride > self.frameskip:
            raise ValueError("Reference history_stride must not exceed frameskip")


def environment(config, *, batch_size=1, observation="rgb"):
    return make(
        "ObstacleArm-v0",
        batch_size=batch_size,
        observation=observation,
        size=config.size,
        quality="low",
        max_episode_steps=config.steps,
        goal_marker=config.goal_marker,
    )


def task(seed):
    """Seeded starts and reachable target poses; native tolerances are unchanged."""
    rng = np.random.default_rng(seed)
    start = [rng.uniform(-0.15, 0), rng.uniform(-0.1, 0.1), 0, 0]
    goal = [rng.uniform(-1.15, -0.55), rng.uniform(0.75, 1.35), 0, 0]
    return torch.tensor(start, dtype=torch.float32), torch.tensor(goal, dtype=torch.float32)


def expert(env, *, gain=10.0):
    """Gravity-compensated PD collector; privileged physics stays outside LeWAM."""
    commands = []
    for data, goal in zip(env.data, env.goal.numpy()):
        torque = data.qfrc_bias + gain * (goal[:2] - data.qpos) - data.qvel
        commands.append(torque / env.model.actuator_gear[:, 0])
    return torch.tensor(np.stack(commands), dtype=torch.float32).clamp(-1, 1)


def demonstration_task(seed, config):
    """Mix standard approaches with seeded near-goal corrections for collection.

    Evaluation always uses ``task(seed)``. Recovery starts broaden training and
    validation coverage without changing task physics or native success.
    """
    start, goal = task(seed)
    rng = np.random.default_rng(seed + 1000000)
    if rng.random() < config.recovery_fraction:
        start[:2] = goal[:2] + torch.tensor(rng.uniform(-0.35, 0.35, 2), dtype=start.dtype)
    if config.diverse_motion:
        mode = seed % 4
        if mode == 1:  # approach from the other direction
            start, goal = goal.clone(), start.clone()
        elif mode == 2:  # recover on either side of the destination
            start[:2] = goal[:2] + torch.tensor(rng.uniform(-0.35, 0.35, 2), dtype=start.dtype)
            start[2:] = torch.tensor(rng.uniform(-0.5, 0.5, 2), dtype=start.dtype)
        elif mode == 3:  # brake close to the destination, then hold
            start[:2] = goal[:2] + torch.tensor(rng.uniform(-0.08, 0.08, 2), dtype=start.dtype)
            start[2:] = torch.tensor(rng.uniform(-1.5, 1.5, 2), dtype=start.dtype)
    return start, goal


def collect(config, seeds):
    """Successful, goal-directed trajectories; no failures are silently discarded."""
    records = []
    with environment(config) as env:
        for index, seed in enumerate(seeds):
            start, goal = demonstration_task(seed, config)
            env.goal = goal[None]
            obs = env.reset(start[None])
            pixels, states, actions = [obs[0]], [env.state[0].clone()], []
            for _ in range(config.steps):
                action = expert(env, gain=config.expert_gain)
                obs = env.step(action)
                pixels.append(obs[0])
                states.append(env.state[0].clone())
                actions.append(action[0])
            if not env.success(env.state, env.goal).item():
                raise RuntimeError(f"Scripted feasibility failed for seed {seed}; do not train on invalid labels")
            records.append(
                {
                    "pixels": (torch.stack(pixels) * 255).round().byte(),
                    "states": torch.stack(states),
                    "actions": torch.stack(actions),
                    "goal_pixels": (obs[0] * 255).round().byte()
                    if config.reference
                    else (env.render(goal[None])[0] * 255).round().byte(),
                    "goal": goal,
                }
            )
            if (index + 1) % 16 == 0:
                print(f"Collected {index + 1}/{len(seeds)} verified demonstrations", flush=True)
    return {key: torch.stack([r[key] for r in records]) for key in records[0]}


def normalize_pixels(pixels, config):
    """ImageNet normalization used by the reference, or the archived v1 protocol."""
    if not config.reference:
        return pixels * 2 - 1
    mean = pixels.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = pixels.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    return (pixels - mean) / std


def windows(data, config, count, generator, *, episode_ids=None, start_steps=None):
    """Pad initial history only; balance early motion with all trajectory phases."""
    ids = torch.randint(len(data["pixels"]), (count,), generator=generator) if episode_ids is None else episode_ids
    if config.reference:
        # Start uniformly, including the tail: reference pads unavailable actions.
        starts = torch.randint(config.steps - config.frameskip + 1, (count,), generator=generator)
        if config.launch_fraction:
            launch = torch.rand(count, generator=generator) < config.launch_fraction
            starts[launch] = torch.randint(
                min(5, config.steps - config.frameskip + 1), (int(launch.sum()),), generator=generator
            )
        if start_steps is not None:
            starts = start_steps
        if (
            ids.shape != (count,)
            or starts.shape != (count,)
            or (starts < 0).any()
            or (starts > config.steps - config.frameskip).any()
        ):
            raise ValueError("Explicit windows need one valid episode and start index per row")
        history_times = starts[:, None] + torch.arange(1 - config.history, 1) * config.history_stride
        future_times = starts[:, None] + config.frameskip
        action_times = starts[:, None] + torch.arange(config.chunk_size)
        valid = action_times < config.steps
        goal_images = data["goal_pixels"][ids]
        goal_inputs = {}
        if config.goal_mode == "sampled":
            horizon = torch.randint(1, config.goal_horizon + 1, (count,), generator=generator)
            horizon = torch.minimum(horizon, (config.steps - starts) // config.frameskip)
            goal_images = data["pixels"][ids, starts + horizon * config.frameskip]
            goal_inputs["remaining"] = horizon.float() / config.goal_horizon
        return {
            **goal_inputs,
            "history_pixels": normalize_pixels(
                data["pixels"][
                    ids[:, None], torch.where(history_times < 0, starts[:, None] % config.history_stride, history_times)
                ].float()
                / 255,
                config,
            ),
            "future_pixels": normalize_pixels(data["pixels"][ids[:, None], future_times].float() / 255, config),
            "actions": data["actions"][ids[:, None], action_times.clamp_max(config.steps - 1)],
            "goal_pixels": normalize_pixels(goal_images.float() / 255, config),
            "history_pad": history_times < 0,
            "action_valid": valid,
            "state_valid": torch.ones(count, 1, dtype=torch.bool),
            "state": data["states"][ids, starts],
            "goal": data["goal"][ids],
        }
    starts = torch.randint(config.steps - config.chunk_size + 1, (count,), generator=generator)
    early = torch.rand(count, generator=generator) < 0.5
    starts[early] = torch.randint(
        min(20, config.steps - config.chunk_size + 1), (int(early.sum()),), generator=generator
    )
    times = starts[:, None] + torch.arange(-config.history + 1, config.chunk_size + 1)
    indices = times.clamp_min(0)
    actions = data["actions"][ids[:, None], indices[:, :-1]]
    actions = actions.masked_fill((times[:, :-1] < 0)[..., None], 0)
    return {
        "pixels": data["pixels"][ids[:, None], indices].float() / 127.5 - 1,
        "actions": actions,
        "goal_pixels": data["goal_pixels"][ids].float() / 127.5 - 1,
        "state": data["states"][ids, starts],
        "goal": data["goal"][ids],
    }


def make_model(config, *, action_mean=None, action_std=None):
    if config.reference:
        from torch_world.models.jepa import LeWAMEncoder

        return LeWAM(
            LeWAMEncoder(),
            2,
            goal_type=config.goal_mode,
            history=config.history,
            chunk_size=config.chunk_size,
            frameskip=config.frameskip,
            num_states=1,
            feature_dim=64,
            projector_norm="batch",
            action_mean=torch.zeros(2) if action_mean is None else action_mean,
            action_std=torch.ones(2) if action_std is None else action_std,
            dropout=0.1,
            flow_steps=config.flow_steps,
        )
    encoder = nn.Sequential(
        nn.Conv2d(3, 16, 5, 2, 2),
        nn.GroupNorm(4, 16),
        nn.GELU(),
        nn.Conv2d(16, 32, 3, 2, 1),
        nn.GroupNorm(4, 32),
        nn.GELU(),
        nn.Conv2d(32, 32, 3, 2, 1),
        nn.GELU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(512, 64),
    )
    return LeWAM(
        encoder,
        2,
        embed_dim=64,
        history=config.history,
        chunk_size=config.chunk_size,
        dim=64,
        depth=2,
        num_heads=4,
        mlp_dim=256,
        flow_steps=config.flow_steps,
    )


class StatePolicy(nn.Module):
    """Privileged diagnostic baseline; never used by the image policy."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, state, goal):
        return self.net(torch.cat((state[:, :2], state[:, 2:] / 5, goal[:, :2]), -1)).clamp(-1, 1)


def _inputs(batch):
    return {key: value for key, value in batch.items() if key not in ("state", "goal")}


def training_loss(model, batch, **kwargs):
    call = model.forward_windows if "history_pixels" in batch else model
    return call(**_inputs(batch), **kwargs)


def fit(train, validation, config, output, device, *, initial_checkpoint=None):
    # Initialization must not depend on whether demonstrations were collected or loaded.
    torch.manual_seed(config.seed)
    stats = {}
    if config.reference:
        flat = train["actions"].flatten(0, 1)
        std = flat.std(0)
        stats = {"action_mean": flat.mean(0), "action_std": torch.where(std == 0, 1, std)}
    model, oracle = make_model(config, **stats).to(device), StatePolicy().to(device)
    if initial_checkpoint is not None:
        initial_checkpoint = Path(initial_checkpoint)
        model.load_state_dict(torch.load(initial_checkpoint, map_location=device, weights_only=True)["model"])
        oracle.load_state_dict(
            torch.load(initial_checkpoint.parent / "state-policy.pt", map_location=device, weights_only=True)["model"]
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.5e-4 if config.reference else 3e-4, weight_decay=1e-4 if config.reference else 0.01
    )
    action_index = 0 if config.reference else config.history - 1
    oracle_optimizer = torch.optim.AdamW(oracle.parameters(), lr=3e-4)
    rng = torch.Generator().manual_seed(config.seed)
    validation_batches = []
    phases = (
        [step for step in (0, 1, 2, 3, 4, 5, 20, 50, 90) if step <= config.steps - config.frameskip]
        if config.phase_validation
        else [None]
    )
    for phase in phases:
        count = len(validation["pixels"]) if phase is not None else config.batch_size
        explicit = (
            {} if phase is None else {"episode_ids": torch.arange(count), "start_steps": torch.full((count,), phase)}
        )
        batch = windows(validation, config, count, torch.Generator().manual_seed(1234), **explicit)
        validation_batches.append({k: v.to(device) for k, v in batch.items()})
    curve, best_flow, best_state = [], float("inf"), float("inf")
    for update in range(1, config.updates + 1):
        model.train()
        oracle.train()
        batch = {k: v.to(device) for k, v in windows(train, config, config.batch_size, rng).items()}
        if config.reference:
            warmup = max(1, config.updates // 10)
            factor = (
                min(update / warmup, 1.0)
                * 0.5
                * (1 + np.cos(np.pi * max(0, update - warmup) / max(1, config.updates - warmup)))
            )
            optimizer.param_groups[0]["lr"] = 1.5e-4 * factor
        losses = training_loss(model, batch)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0 if config.reference else 10.0, error_if_nonfinite=True)
        optimizer.step()
        oracle_loss = (oracle(batch["state"], batch["goal"]) - batch["actions"][:, action_index]).square().mean()
        oracle_optimizer.zero_grad(set_to_none=True)
        oracle_loss.backward()
        oracle_optimizer.step()
        if update % config.validation_interval == 0 or update == config.updates:
            model.eval()
            oracle.eval()
            with torch.no_grad():
                validation_flow, state_errors = [], []
                for index, val in enumerate(validation_batches):
                    checked = training_loss(
                        model, val, generator=torch.Generator(device=device).manual_seed(1234 + index)
                    )
                    validation_flow.append(checked["action_loss"])
                    state_errors.append(
                        (oracle(val["state"], val["goal"]) - val["actions"][:, action_index]).square().mean()
                    )
                flow_loss = torch.stack(validation_flow).mean()
                state_mse = float(torch.stack(state_errors).mean())
            row = {
                "update": update,
                **{k: float(v.detach()) for k, v in losses.items()},
                "validation_flow": float(flow_loss),
                "validation_state_mse": state_mse,
            }
            if not all(np.isfinite(v) for v in row.values()):
                raise RuntimeError("Nonfinite training/validation metrics")
            curve.append(row)
            if row["validation_flow"] < best_flow:
                best_flow = row["validation_flow"]
                torch.save(
                    {"config": asdict(config), "model": model.state_dict(), "update": update}, output / "model.pt"
                )
            if state_mse < best_state:
                best_state = state_mse
                torch.save({"model": oracle.state_dict(), "update": update}, output / "state-policy.pt")
            print(json.dumps(row), flush=True)
    checkpoint = torch.load(output / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    state_checkpoint = torch.load(output / "state-policy.pt", map_location=device, weights_only=True)
    oracle.load_state_dict(state_checkpoint["model"])
    return model.eval(), oracle.eval(), curve, {"lewam": checkpoint["update"], "state": state_checkpoint["update"]}


@torch.no_grad()
def sample_control(model, history, goal, samples, generator, *, history_pad=None, return_chunk=False, remaining=None):
    """Monte Carlo mean of complete flow samples, not zero-noise initialization."""
    b = history.shape[0]
    z = model.encode(history)
    actions = model.sample_actions(
        z.repeat_interleave(samples, 0),
        goal.repeat_interleave(samples, 0),
        remaining=None if remaining is None else remaining.repeat_interleave(samples, 0),
        generator=generator,
        history_pad=None if history_pad is None else history_pad.repeat_interleave(samples, 0),
    )
    chunks = actions.reshape(b, samples, model.chunk_size, 2).mean(1)
    return chunks if return_chunk else chunks[:, 0]


@torch.no_grad()
def diagnose(model, validation, config, device):
    batch = {
        k: v.to(device)
        for k, v in windows(
            validation, config, config.batch_size if config.reference else 128, torch.Generator().manual_seed(44)
        ).items()
    }
    correct = training_loss(model, batch, generator=torch.Generator(device=device).manual_seed(7))
    shuffled = training_loss(
        model,
        {**batch, "goal_pixels": batch["goal_pixels"].roll(1, 0)},
        generator=torch.Generator(device=device).manual_seed(7),
    )
    h = batch["history_pixels"] if config.reference else batch["pixels"][:, : config.history]
    g = model.encode(batch["goal_pixels"][:, None])[:, 0]
    sampled = sample_control(
        model,
        h,
        g,
        config.mean_samples,
        torch.Generator(device=device).manual_seed(7),
        history_pad=batch.get("history_pad"),
        remaining=batch.get("remaining"),
    )
    wrong = sample_control(
        model,
        h,
        g.roll(1, 0),
        config.mean_samples,
        torch.Generator(device=device).manual_seed(7),
        history_pad=batch.get("history_pad"),
        remaining=batch.get("remaining"),
    )
    target = batch["actions"][:, 0 if config.reference else config.history - 1]
    return {
        "correct_goal_flow_loss": float(correct["action_loss"]),
        "shuffled_goal_flow_loss": float(shuffled["action_loss"]),
        "mean_action_mse": float((sampled - target).square().mean()),
        "zero_action_mse": float(target.square().mean()),
        "goal_change_rms": float((sampled - wrong).square().mean().sqrt()),
        "latent_std": float(model.encode(h).flatten(0, 1).std(0).mean()),
    }


@torch.no_grad()
def diagnose_phases(model, validation, config, device, *, steps=None):
    """Physical first-action errors at fixed phases, including padded initial history.

    Uses every validation episode, with no test rollouts or privileged inputs to
    the visual policy. Aggregate random-window loss can miss launch failures.
    """
    if steps is None:
        steps = tuple(step for step in (0, 5, 20, 50, 90) if step < config.steps)
    goal = model.encode(normalize_pixels(validation["goal_pixels"].to(device).float() / 255, config)[:, None])[:, 0]
    results = []
    for step in steps:
        if not 0 <= step < config.steps:
            raise ValueError("Probe steps must lie inside the episode")
        times = step + torch.arange(1 - config.history, 1) * config.history_stride
        indices = torch.where(times < 0, step % config.history_stride, times)
        images = normalize_pixels(validation["pixels"][:, indices].to(device).float() / 255, config)
        pad = (times < 0)[None].expand(len(images), -1).to(device)
        predicted = sample_control(
            model,
            images,
            goal,
            config.mean_samples,
            torch.Generator(device=device).manual_seed(7),
            history_pad=pad if config.reference else None,
        )
        target = validation["actions"][:, step].to(device)
        results.append(
            {
                "step": step,
                "episodes": len(images),
                "action_mse": float((predicted - target).square().mean()),
                "predicted_mean": predicted.mean(0).tolist(),
                "expert_mean": target.mean(0).tolist(),
            }
        )
    return results


@torch.no_grad()
def probe_state(model, train, validation, config, device):
    """Frozen linear probes: fit on training episodes, measure on validation only.

    These probes do not modify LeWAM or supply state to its policy. R2 below zero
    means worse than predicting the validation mean; RMSE keeps physical units.
    A linear probe cannot establish that information is absent nonlinearly.
    """

    def features(data, count, seed):
        rng = torch.Generator().manual_seed(seed)
        codes, states = [], []
        for offset in range(0, count, 64):
            batch = windows(data, config, min(64, count - offset), rng)
            images = batch["history_pixels"] if config.reference else batch["pixels"][:, : config.history]
            z = model.encode(images.to(device))
            codes.append(z.cpu().double())
            states.append(batch["state"].double())
        return torch.cat(codes), torch.cat(states)

    x, y = features(train, 4096, 765)
    test_x, test_y = features(validation, 512, 876)
    results = {}
    for name, a, b in (("single_frame", x[:, -1], test_x[:, -1]), ("history", x.flatten(1), test_x.flatten(1))):
        mean, scale = a.mean(0), a.std(0).clamp_min(1e-5)
        a, b = (a - mean) / scale, (b - mean) / scale
        target_mean = y.mean(0)
        gram = a.T @ a / len(a) + 0.01 * torch.eye(a.shape[1], dtype=a.dtype)
        weights = torch.linalg.solve(gram, a.T @ (y - target_mean) / len(a))
        predictions = b @ weights + target_mean
        mse = (predictions - test_y).square().mean(0)
        results[name] = {
            "rmse": mse.sqrt().tolist(),
            "r2": (1 - mse / test_y.var(0, unbiased=False).clamp_min(1e-8)).tolist(),
        }
    results["coordinates"] = ["shoulder_rad", "elbow_rad", "shoulder_rad_per_s", "elbow_rad_per_s"]
    return results


def evaluate(model, oracle, config, seeds, output, device, *, execution_interval=None, controllers=None):
    """Native success includes pose tolerance and settling; no relaxed criterion."""
    from PIL import Image

    execution_interval = (
        (config.execution_interval or config.frameskip) if execution_interval is None else execution_interval
    )
    if not 1 <= execution_interval <= config.chunk_size:
        raise ValueError("execution_interval must be between one and chunk_size")
    available = ("lewam_single", "lewam_mean", "shuffled_goal_mean", "state_policy", "expert", "zero", "random")
    controllers = available if controllers is None else tuple(controllers)
    if not controllers or not set(controllers) <= set(available):
        raise ValueError("Unknown or empty controller selection")
    starts, goals = map(torch.stack, zip(*(task(seed) for seed in seeds)))
    results = {}
    for name in controllers:
        visual = name in ("lewam_single", "lewam_mean", "shuffled_goal_mean")
        with environment(config, batch_size=len(seeds), observation="rgb" if visual else "state") as env:
            env.goal = goals.clone()
            obs = env.reset(starts)
            first_hit = torch.full((len(seeds),), -1, dtype=torch.long)
            initial = env.success(env.state, env.goal)
            first_hit[initial] = 0
            best = env.distance(env.state, env.goal)
            if visual:
                history = [normalize_pixels(obs.to(device), config)]
                with torch.no_grad():
                    target = model.encode(normalize_pixels(env.render(goals).to(device), config)[:, None])[:, 0]
                if name == "shuffled_goal_mean":
                    target = target.roll(1, 0)
            rng = torch.Generator(device=device).manual_seed(999)
            random_rng = torch.Generator().manual_seed(999)
            images, goal_image = [], None
            trace = {"states": [env.state.tolist()], "actions": [], "expert_actions": []}
            for step in range(config.steps):
                if visual:
                    count = 1 if name == "lewam_single" else config.mean_samples
                    if step % execution_interval == 0:
                        times = [step + i * config.history_stride for i in range(1 - config.history, 1)]
                        frames = torch.stack([history[max(step % config.history_stride, t)] for t in times], 1)
                        pad = torch.tensor([t < 0 for t in times], device=device)[None].expand(len(seeds), -1)
                        cached = sample_control(
                            model,
                            frames,
                            target,
                            count,
                            rng,
                            history_pad=pad if config.reference else None,
                            return_chunk=True,
                        ).cpu()
                    action = cached[:, step % execution_interval]
                elif name == "state_policy":
                    with torch.no_grad():
                        action = oracle(env.state.to(device), goals.to(device)).cpu()
                elif name == "expert":
                    action = expert(env, gain=config.expert_gain)
                else:
                    action = (
                        torch.zeros(len(seeds), 2)
                        if name == "zero"
                        else torch.rand(len(seeds), 2, generator=random_rng) * 2 - 1
                    )
                trace["actions"].append(action.tolist())
                trace["expert_actions"].append(expert(env, gain=config.expert_gain).tolist())
                obs = env.step(action)
                trace["states"].append(env.state.tolist())
                hits = env.success(env.state, env.goal)
                first_hit[(first_hit < 0) & hits] = step + 1
                best = torch.minimum(best, env.distance(env.state, env.goal))
                if visual:
                    history.append(normalize_pixels(obs.to(device), config))
                    if name == "lewam_mean":
                        if goal_image is None:
                            goal_image = env.render(goals)[0]
                        pair = torch.cat((obs[0], goal_image), -1).permute(1, 2, 0)
                        images.append(Image.fromarray((pair.numpy() * 255).round().astype(np.uint8)).resize((512, 256)))
            final = env.success(env.state, env.goal)
            distances = env.distance(env.state, env.goal)
            results[name] = [
                {
                    "seed": seed,
                    "success": bool(first_hit[i] >= 0),
                    "initial_success": bool(initial[i]),
                    "first_hit_step": int(first_hit[i]),
                    "final_success": bool(final[i]),
                    "best_normalized_distance": float(best[i]),
                    "final_normalized_distance": float(distances[i]),
                }
                for i, seed in enumerate(seeds)
            ]
            (output / f"trace-{name}.json").write_text(
                json.dumps({"seeds": list(seeds), "goals": goals.tolist(), **trace}) + "\n"
            )
            if images:
                images[0].save(output / "rollout.gif", save_all=True, append_images=images[1:], duration=50, loop=0)
        print(
            f"{name}: {sum(t['success'] for t in results[name])}/{len(seeds)} first-hit; "
            f"{sum(t['final_success'] for t in results[name])}/{len(seeds)} final",
            flush=True,
        )
    return results


def run_arm(output, config=None, *, device=None, data_path=None, initial_checkpoint=None):
    config = config or ArmConfig()
    if config.goal_mode != "terminal":
        raise ValueError("Use run_goal_reaching for sampled-goal training and evaluation")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    torch.manual_seed(config.seed)
    started = time.perf_counter()
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = {}
    for relative in (
        "benchmarks/lewam_arm.py",
        "benchmarks/lewam_arm_diagnostics.py",
        "models/jepa/_lewam.py",
        "models/jepa/_lewam_encoder.py",
        "envs/physics_scenes.py",
        "envs/physics_tasks.py",
        "envs/mujoco.py",
    ):
        content = (source_root / relative).read_bytes()
        source_hashes[relative] = hashlib.sha256(content).hexdigest()
        frozen = output / "source" / relative
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(content)
    seeds = {
        name: [config.seed + offset + i for i in range(count)]
        for name, offset, count in (
            ("train", 0, config.train_episodes),
            ("validation", 10000, config.validation_episodes),
            ("test", config.test_seed_offset, config.test_episodes),
        )
    }
    if len({seed for split in seeds.values() for seed in split}) != sum(map(len, seeds.values())):
        raise ValueError("Episode seeds overlap")
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2))
    # Test demonstrations are not collected or used to train/select checkpoints.
    if data_path is None:
        data = {name: collect(config, seeds[name]) for name in ("train", "validation")}
        torch.save(data, output / "episodes.pt")
    else:
        data_path = Path(data_path).resolve()
        source_config = ArmConfig(**json.loads((data_path.parent / "config.json").read_text()))
        for key in (
            "seed",
            "steps",
            "size",
            "train_episodes",
            "validation_episodes",
            "reference",
            "expert_gain",
            "recovery_fraction",
            "goal_marker",
            "diverse_motion",
        ):
            if getattr(config, key) != getattr(source_config, key):
                raise ValueError(f"Cached demonstrations disagree on {key}")
        data = torch.load(data_path, map_location="cpu", weights_only=True)
        (output / "data-source.json").write_text(json.dumps({"path": str(data_path)}))
    from torch_world.benchmarks.lewam_arm_diagnostics import audit_demonstrations

    audit = audit_demonstrations(data["train"], config)
    (output / "environment-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if any(
        row["max_state_error"] > 1e-5
        or row["max_pixel_error_uint8"] > 1
        or row["goal_render_state_error"] > 1e-7
        or not row["final_success"]
        for row in audit["replay"]
    ):
        raise RuntimeError("Demonstration replay failed; inspect environment-audit.json before training")
    model, oracle, curve, selected = fit(
        data["train"], data["validation"], config, output, device, initial_checkpoint=initial_checkpoint
    )
    diagnostics = diagnose(model, data["validation"], config, device)
    phase_diagnostics = diagnose_phases(model, data["validation"], config, device) if config.reference else []
    representation_probe = (
        {} if config.reference else probe_state(model, data["train"], data["validation"], config, device)
    )
    evaluation = evaluate(model, oracle, config, seeds["test"], output, device)
    import mujoco

    result = {
        "protocol": "lewam-owned-arm-reference-bc-v2" if config.reference else "lewam-owned-arm-v1",
        "reference_commit": "f57659fb26858082975794d37a8f3d380ddc1fa7" if config.reference else None,
        "environment": "ObstacleArm-v0",
        "source_sha256": source_hashes,
        "environment_audit": audit,
        "initial_checkpoint_sha256": None
        if initial_checkpoint is None
        else hashlib.sha256(Path(initial_checkpoint).read_bytes()).hexdigest(),
        "config": asdict(config),
        "episode_seeds": seeds,
        "selected_updates": selected,
        "curve": curve,
        "diagnostics": diagnostics,
        "phase_diagnostics": phase_diagnostics,
        "representation_probe": representation_probe,
        "evaluation": evaluation,
        "host": platform.node(),
        "device": device,
        "accelerator": torch.cuda.get_device_name() if device == "cuda" else device,
        "versions": {"torch": torch.__version__, "mujoco": mujoco.__version__},
        "checkpoint_sha256": hashlib.sha256((output / "model.pt").read_bytes()).hexdigest(),
        "seconds": time.perf_counter() - started,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_recovery_curriculum(output, config, *, device=None):
    """Reproduce the four-stage recovery experiment, keeping every stage's report.

    Final-stage ``config`` describes the recovery dataset and fresh evaluation
    seeds. Earlier stages use the original evaluation tasks for diagnostics.
    Each warm start retains the source action normalization and resets AdamW.
    """
    if not config.reference or config.recovery_fraction <= 0:
        raise ValueError("Recovery curriculum requires the reference model and recovery starts")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    initial = replace(
        config,
        updates=2000,
        history_stride=5,
        execution_interval=5,
        expert_gain=10,
        recovery_fraction=0,
        launch_fraction=0,
        phase_validation=False,
        test_seed_offset=20000,
    )
    coverage = replace(initial, updates=500, launch_fraction=0.5, phase_validation=True)
    smooth = replace(config, updates=1000, recovery_fraction=0, test_seed_offset=20000)
    run_arm(output / "initial", initial, device=device)
    run_arm(
        output / "coverage",
        coverage,
        device=device,
        data_path=output / "initial/episodes.pt",
        initial_checkpoint=output / "initial/model.pt",
    )
    run_arm(output / "smooth", smooth, device=device, initial_checkpoint=output / "coverage/model.pt")
    return run_arm(output / "recovery", config, device=device, initial_checkpoint=output / "smooth/model.pt")


def evaluate_checkpoint(checkpoint_path, output, *, execution_interval=1, device=None):
    """Compare execution cadence without retraining or selecting on test results.

    Reuses the completed run's exact held-out seeds and checkpoint. The three
    visual policies share random draws and target assignments with the main run.
    Outputs are separate, so the original evaluation cannot be overwritten.
    """
    checkpoint_path, output = Path(checkpoint_path), Path(output)
    source = json.loads((checkpoint_path.parent / "results.json").read_text())
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if digest != source["checkpoint_sha256"]:
        raise ValueError("Checkpoint does not match the completed training report")
    config = ArmConfig(**source["config"])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = make_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    started = time.perf_counter()
    result = {
        "protocol": "lewam-owned-arm-fixed-checkpoint-cadence",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint_sha256": digest,
        "execution_interval": execution_interval,
        "test_seeds": source["episode_seeds"]["test"],
        "evaluation": evaluate(
            model,
            None,
            config,
            source["episode_seeds"]["test"],
            output,
            device,
            execution_interval=execution_interval,
            controllers=("lewam_single", "lewam_mean", "shuffled_goal_mean"),
        ),
        "seconds": time.perf_counter() - started,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--updates", type=int, default=4000)
    parser.add_argument(
        "--reference", action="store_true", help="Reference-aligned terminal-goal BC, on our environment"
    )
    parser.add_argument("--initialize-from", type=Path, help="Warm-start weights, with a fresh optimizer and schedule")
    parser.add_argument("--launch-fraction", type=float, default=0.0)
    parser.add_argument("--phase-validation", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="Evaluate a completed checkpoint instead of training")
    parser.add_argument("--execution-interval", type=int, default=None)
    parser.add_argument("--history-stride", type=int, default=5)
    parser.add_argument("--expert-gain", type=float, default=10.0)
    parser.add_argument("--recovery-fraction", type=float, default=0.0)
    parser.add_argument("--test-seed-offset", type=int, default=20000)
    parser.add_argument("--data", type=Path, help="Reuse verified episodes.pt with matching config.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.checkpoint is not None:
        evaluate_checkpoint(
            args.checkpoint,
            args.output,
            execution_interval=1 if args.execution_interval is None else args.execution_interval,
            device=args.device,
        )
        return
    options = (
        {
            "size": 224,
            "history": 2,
            "chunk_size": 10,
            "frameskip": 5,
            "history_stride": args.history_stride,
            "flow_steps": 8,
        }
        if args.reference
        else {}
    )
    run_arm(
        args.output,
        ArmConfig(
            updates=args.updates,
            reference=args.reference,
            batch_size=args.batch_size,
            launch_fraction=args.launch_fraction,
            phase_validation=args.phase_validation,
            execution_interval=args.execution_interval,
            expert_gain=args.expert_gain,
            recovery_fraction=args.recovery_fraction,
            test_seed_offset=args.test_seed_offset,
            **options,
        ),
        device=args.device,
        data_path=args.data,
        initial_checkpoint=args.initialize_from,
    )


def show_results(result):
    """Plot action learning and native first-hit/final success separately."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    curve = result["curve"]
    validation_label = "Phase validation mean" if result["config"].get("phase_validation") else "Fixed validation batch"
    for key, label in (("action_loss", "Training batch"), ("validation_flow", validation_label)):
        axes[0].plot([r["update"] for r in curve], [r[key] for r in curve], label=label)
    axes[0].set(xlabel="Update", ylabel="Action flow loss", title="Action flow loss")
    axes[0].legend()
    names = list(result["evaluation"])
    x = np.arange(len(names))
    for offset, key, label in ((-0.18, "success", "First hit"), (0.18, "final_success", "Final state")):
        rates = [100 * np.mean([r[key] for r in result["evaluation"][name]]) for name in names]
        axes[1].bar(x + offset, rates, width=0.36, label=label)
    axes[1].set(
        ylim=(0, 105), ylabel="Native success (%)", title="Identical held-out tasks", xticks=x, xticklabels=names
    )
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
