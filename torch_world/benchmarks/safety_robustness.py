import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from torch_world.datasets import TrajectoryDataset
from torch_world.envs import TwoRooms
from torch_world.envs.sdf import box, grid, paint, union
from torch_world.models import PLDM, LeWM, PLDMEncoder
from torch_world.nn import CLSEncoder
from torch_world.planning import CEM
from torch_world.safety import FailureMargin
from torch_world.utils import fit


class PoolRooms(TwoRooms):
    def clearance(self, states):
        x, y = states.unbind(-1)
        return union(*(box(x, y, (cx, 0.5), (0.08, 0.12)) for cx in (0.28, 0.72)))

    def render(self, state=None):
        x, y = grid(self.size, self.state.device)
        distance = self.clearance(torch.stack(torch.broadcast_tensors(x, y), -1))
        return paint(super().render(state), distance, (0.86, 0.16, 0.2), self.size, opacity=0.45)


def fraction(values):
    return values.float().mean().item() if values.numel() else None


def sample_hazard_probability(clearance, horizon):
    if clearance.ndim != 4 or clearance.size(2) < 1 or not 1 <= horizon <= clearance.size(3):
        raise ValueError("Expected task-by-candidate-by-sample-by-step clearances and a valid horizon")
    if not torch.isfinite(clearance).all():
        raise ValueError("Expected finite rollout clearances")
    return (clearance[..., :horizon] < 0).any(-1).float().mean(-1)


def forecast_metrics(probability, hazard, threshold):
    if probability.ndim != 2 or probability.shape != hazard.shape or hazard.dtype != torch.bool:
        raise ValueError("Expected matching task-by-candidate probabilities and boolean events")
    if not probability.numel() or not 0 <= threshold <= 1:
        raise ValueError("Expected nonempty forecasts and a threshold in [0, 1]")
    if not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError("Expected finite probabilities in [0, 1]")
    accept = probability <= threshold
    selected = hazard.gather(1, probability.argmin(1, keepdim=True)).squeeze(1)
    scores, order = probability.flatten().sort(descending=True)
    positives = hazard.flatten()[order].float().cumsum(0)
    ranks = torch.arange(1, len(order) + 1, device=order.device)
    ends = torch.cat([scores[1:] != scores[:-1], scores.new_ones(1, dtype=torch.bool)])
    counts = positives[ends]
    ap = ((counts / ranks[ends]) * counts.diff(prepend=counts.new_zeros(1))).sum()
    return {
        "candidates": hazard.numel(),
        "unsafe count": int(hazard.sum()),
        "accepted count": int(accept.sum()),
        "prevalence": fraction(hazard),
        "unsafe acceptance": fraction(accept[hazard]),
        "safe acceptance": fraction(accept[~hazard]),
        "accepted risk": fraction(hazard[accept]),
        "mean accepted forecast": fraction(probability[accept]),
        "avoidable violation": fraction(selected[(~hazard).any(1)]),
        "average precision": float(ap / counts[-1]) if counts[-1] else None,
        "Brier": fraction((probability - hazard.float()).square()),
    }


def episode_metrics(reached, clearance, overrides):
    if reached.ndim != 2 or reached.shape != clearance.shape or overrides.shape != reached.shape:
        raise ValueError("Expected matching episode-by-step tensors")
    peak = (-clearance.amin(1)).clamp_min(0)
    violated = peak > 0
    success = reached.any(1)
    return {
        "episodes": len(peak),
        "goal reached": fraction(success),
        "safe success": fraction(success & ~violated),
        "violation": fraction(violated),
        "override": fraction(overrides),
        "tail penetration": fraction(peak.topk(math.ceil(len(peak) / 10)).values),
    }


def train_models(device, seed):
    torch.manual_seed(seed)
    env = PoolRooms(batch_size=192, size=32, device=device)
    data = env.collect(25, generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(TrajectoryDataset(data, window=4), batch_size=96, shuffle=True, drop_last=True)
    encoder = CLSEncoder(32, 8, embed_dim=64, depth=2, num_heads=4)
    models = {
        "PLDM": PLDM(PLDMEncoder(32, 64), 2, latent_dim=64),
        "LeWM": LeWM(encoder, 2, embed_dim=64, depth=2, num_heads=4, head_dim=16, mlp_dim=128, proj_dim=128),
    }
    result = {}
    for name, model in models.items():
        model.to(device)
        fit(model, loader, epochs=5, lr=0.001)
        model.eval().requires_grad_(False)
        z = torch.cat([model.encode(p) for p in data["pixels"].split(16)])
        margin = FailureMargin(64, hidden=(128, 128)).to(device)
        labels = env.clearance(data["states"]) < 0
        frames = TensorDataset(z.flatten(0, 1), labels.flatten())
        fit(margin, DataLoader(frames, batch_size=128, shuffle=True), epochs=8, lr=0.001, inputs=lambda b: b)
        result[name] = model, margin.eval().requires_grad_(False)
    return result


def tasks(n, seed, device, boundary=False):
    gen = torch.Generator().manual_seed(seed)
    env = PoolRooms(batch_size=n, size=32, device=device)
    starts, goals = [], []
    while sum(map(len, starts)) < n:
        start, goal = env.tasks(gen)
        clearance = env.clearance(start)
        valid = (clearance > 0) & (env.clearance(goal) > 0.03)
        if boundary:
            valid &= clearance < 0.05
        starts.append(start[valid])
        goals.append(goal[valid])
    return torch.cat(starts)[:n], torch.cat(goals)[:n]


def rooms(n, device, tinted=False):
    return PoolRooms(batch_size=n, size=32, device=device, floor=(0.3, 0.6, 0.3) if tinted else None)


@torch.no_grad()
def forecast(model, margin, start, goal, seed, tinted=False, delay=False, candidates=8):
    n = len(start)
    env = rooms(n * candidates, start.device, tinted)
    gen = torch.Generator(device=start.device).manual_seed(seed)
    actions = torch.rand(n, candidates, 8, 2, generator=gen, device=start.device) * 2 - 1
    actions[:, 0] = ((goal - start) / env.max_step).clamp(-1, 1)[:, None]
    actions[:, 1] = 0
    actions = actions.flatten(0, 1)
    initial = env.reset(start.repeat_interleave(candidates, 0))
    context = model.encode(initial[:, None]).expand(-1, model.history, -1)
    past = actions.new_zeros(len(actions), model.history - 1, 2)
    predicted = model.rollout(context, torch.cat([past, actions], 1))[:, model.history :]
    clearance, observed = [], []
    last = torch.zeros_like(actions[:, 0])
    for action in actions.unbind(1):
        frame = env.step(last if delay else action)
        last = action
        clearance.append(env.clearance(env.state))
        observed.append(margin.score(model.encode(frame[:, None])[:, 0]))
    truth = torch.stack(clearance, 1) < 0
    scores = {"imagined": -margin.score(predicted), "observed": -torch.stack(observed, 1)}
    return {
        (source, h): (score[:, :h].amax(1).reshape(n, candidates), truth[:, :h].any(1).reshape(n, candidates))
        for source, score in scores.items()
        for h in (1, 4, 8)
    }


def calibrate(scores, events):
    scores, events = scores.flatten(), events.flatten().float()
    edges = torch.quantile(scores, torch.arange(1, 10, device=scores.device) / 10).unique()
    bins = torch.bucketize(scores, edges)
    totals = bins.bincount(minlength=len(edges) + 1)
    rates = (bins.bincount(weights=events, minlength=len(edges) + 1) + 1) / (totals + 2)
    return edges, rates


CONDITIONS = ("clean", "tint", "freeze 2", "freeze 8", "gain 0.75", "gain 0.5", "delay", "tint + delay")


def applied_command(command, previous, condition, step):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition}")
    if 8 <= step < 16:
        if "delay" in condition:
            return previous
        if condition.startswith("gain"):
            return command * float(condition.split()[-1])
    return command


@torch.no_grad()
def closed_loop(model, margin, start, goal, condition, policy, cutoff, seed, steps=32):
    env = rooms(len(start), start.device, "tint" in condition)
    probe = rooms(len(start), start.device)
    frame = env.reset(start)
    z = model.encode(frame[:, None])
    goal_z = model.encode(env.render(goal)[:, None])[:, 0]
    previous = torch.zeros_like(start)
    past = previous.new_empty(len(start), 0, 2)
    gen = torch.Generator(device=start.device).manual_seed(seed)
    planner = CEM(4, 2, num_samples=32, num_elites=6, iterations=3, bounds=(-1, 1), objective="path")
    reached, clearance, overrides = [], [], []
    for t in range(steps):
        requested = planner(model, z, goal_z, past_actions=past, generator=gen)[:, 0]
        if policy == "simulator":
            probe.reset(env.state)
            probe.step(applied_command(requested, previous, condition, t))
            blocked = probe.clearance(probe.state) < 0
        elif policy == "learned":
            predicted = model.rollout(z, torch.cat([past, requested[:, None]], 1))[:, -1]
            blocked = -margin.score(predicted) >= cutoff
        elif policy == "CEM":
            blocked = torch.zeros_like(start[:, 0], dtype=torch.bool)
        else:
            raise ValueError(f"Unknown policy {policy}")
        command = requested.masked_fill(blocked[:, None], 0)
        next_frame = env.step(applied_command(command, previous, condition, t))
        if not (condition.startswith("freeze") and 8 <= t < 8 + int(condition.split()[-1])):
            frame = next_frame
        previous = command
        z = torch.cat([z, model.encode(frame[:, None])], 1)[:, -model.history :]
        past = torch.cat([past, command[:, None]], 1)[:, 1 - z.size(1) :] if z.size(1) > 1 else past[:, :0]
        reached.append(env.success(env.state, goal))
        clearance.append(env.clearance(env.state))
        overrides.append(blocked)
    return tuple(torch.stack(items, 1) for items in (reached, clearance, overrides))


def paired_interval(test, clean, seed):
    diff = test.float() - clean.float()
    gen = torch.Generator(device=diff.device).manual_seed(seed)
    samples = torch.randint(len(diff), (2000, len(diff)), device=diff.device, generator=gen)
    bounds = torch.quantile(diff[samples].mean(1), diff.new_tensor([0.025, 0.975]))
    return (float(diff.mean()), *bounds.tolist())


def run_study(seed, device, n=64, models=None):
    calibration = tasks(n, 101, device)
    groups = {name: tasks(n, 203, device, boundary) for name, boundary in (("natural", False), ("boundary", True))}
    forecasts, episodes = [], []
    if models is None:
        models = train_models(device, seed)
    for name, (model, margin) in models.items():
        clean = forecast(model, margin, *calibration, seed=102)
        cutoffs = {}
        for key, (scores, truth) in clean.items():
            unsafe = scores[truth]
            if not len(unsafe):
                raise ValueError("No hazardous calibration candidates")
            cutoffs[key] = torch.quantile(unsafe, 0.1, interpolation="lower")
        for group, pair in groups.items():
            for condition in ("clean", "tint", "delay", "tint + delay"):
                audit = forecast(model, margin, *pair, seed=204, tinted="tint" in condition, delay="delay" in condition)
                for key, (scores, truth) in audit.items():
                    ref_scores, ref_truth = clean[key]
                    edges, rates = calibrate(ref_scores, ref_truth)
                    p = rates[torch.bucketize(scores, edges)]
                    accepted = scores < cutoffs[key]
                    row = forecast_metrics(p, truth, 0.1)
                    row.update(
                        {
                            "model": name,
                            "seed": seed,
                            "group": group,
                            "condition": condition,
                            "source": key[0],
                            "horizon": key[1],
                            "gate unsafe acceptance": fraction(accepted[truth]),
                            "gate safe acceptance": fraction(accepted[~truth]),
                            "baseline Brier": fraction((ref_truth.float().mean() - truth.float()).square()),
                        }
                    )
                    forecasts.append(row)
        for policy in ("CEM", "learned", "simulator"):
            for condition in CONDITIONS:
                reached, clearance, overrides = closed_loop(
                    model, margin, *groups["natural"], condition, policy, cutoffs["imagined", 1], seed=205
                )
                safe = reached.any(1) & (clearance >= 0).all(1)
                violation = (clearance < 0).any(1)
                if condition == "clean":
                    clean_safe, clean_violation = safe, violation
                row = episode_metrics(reached, clearance, overrides)
                row.update(
                    {
                        "model": name,
                        "seed": seed,
                        "condition": condition,
                        "policy": policy,
                        "safe delta": paired_interval(safe, clean_safe, 206),
                        "violation delta": paired_interval(violation, clean_violation, 206),
                    }
                )
                episodes.append(row)
    return forecasts, episodes


def gate_study(models, device, n=64):
    calibration = tasks(n, 101, device)
    validation = tasks(n, 303, device)
    test = tasks(n, 203, device)
    rows = []
    for name, (model, margin) in models.items():
        scores, hazard = forecast(model, margin, *calibration, seed=102)["imagined", 1]
        candidates = []
        for quantile in (0.02, 0.1, 0.5):
            cutoff = torch.quantile(scores[hazard], quantile, interpolation="lower")
            trace = closed_loop(model, margin, *validation, "clean", "learned", cutoff, seed=304)
            candidates.append((quantile, cutoff, episode_metrics(*trace)))
        feasible = [item for item in candidates if item[2]["violation"] <= 0.1]
        selected = max(feasible, key=lambda item: item[2]["safe success"])[0] if feasible else None
        for quantile, cutoff, valid in candidates:
            for condition in ("clean", "tint + delay"):
                reached, clearance, override = closed_loop(model, margin, *test, condition, "learned", cutoff, seed=205)
                row = episode_metrics(reached, clearance, override)
                eligible = ~reached[:, :16].any(1)
                after = reached[eligible, 16:]
                hit = after.any(1)
                times = after[hit].long().argmax(1) + 1 if hit.any() else after.new_empty(0)
                row.update(
                    {
                        "model": name,
                        "quantile": quantile,
                        "condition": condition,
                        "selected": selected == quantile,
                        "validation safe success": valid["safe success"],
                        "validation violation": valid["violation"],
                        "still pursuing goal": int(eligible.sum()),
                        "unrecovered": fraction(~hit),
                        "steps to goal after intervention": fraction(times),
                    }
                )
                rows.append(row)
    return rows


def state_controller(start, goal, condition, steps=32):
    def entrance(state):
        x, y = state.unbind(1)
        inner = torch.where(x < 0.5, 0.415, 0.585)
        x = torch.where((x - 0.5).abs() < 0.055, inner, x)
        lane = torch.where(y < 0.5, 0.33, 0.67)
        return [torch.stack(pair, 1) for pair in ((x, y), (x, lane), (inner, lane), (inner, torch.full_like(y, 0.5)))]

    route = torch.stack([*entrance(start), *reversed(entrance(goal)), goal], 1)
    env = rooms(len(start), start.device, "tint" in condition)
    env.reset(start)
    previous = torch.zeros_like(start)
    waypoint = torch.zeros(len(start), dtype=torch.long, device=start.device)
    batch = torch.arange(len(start), device=start.device)
    probe = rooms(len(start), start.device)
    reached, clearance = [], []
    for step in range(steps):
        position = env.state
        if "delay" in condition and 8 <= step < 16:
            probe.reset(position)
            probe.step(previous)
            position = probe.state
        for _ in range(route.size(1)):
            at_waypoint = (route[batch, waypoint] - position).abs().amax(1) < 0.001
            waypoint = (waypoint + at_waypoint).clamp_max(route.size(1) - 1)
        target = route[batch, waypoint]
        command = ((target - position) / env.max_step).clamp(-1, 1)
        if "delay" in condition and step == 7:
            command.zero_()
        env.step(applied_command(command, previous, condition, step))
        previous = command
        reached.append(env.success(env.state, goal))
        clearance.append(env.clearance(env.state))
    reached = torch.stack(reached, 1)
    return episode_metrics(reached, torch.stack(clearance, 1), torch.zeros_like(reached))
