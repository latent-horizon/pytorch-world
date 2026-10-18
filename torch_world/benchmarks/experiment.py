"""Shared, deliberately small LeWM experiment used by the environment notebooks.

This is a validation budget, not a reproduction of the paper's training schedule.
"""

import time
from collections import deque

import torch
from torch import nn

from torch_world.benchmarks.artifacts import configuration, save_run
from torch_world.envs import get_spec
from torch_world.envs.registry import environment_scope, is_environment
from torch_world.models import LeWM
from torch_world.planning import CEM


def reachable_states(env, limit=100000):
    """Enumerate reachable nonterminal puzzle states; no expert actions enter training."""
    queue, seen = deque([env.initial]), {env.initial}
    while queue:
        state = queue.popleft()
        s = torch.tensor([state], dtype=torch.float32, device=env.device)
        if env.success(s, env.goal[:1]).item():
            continue
        for direction in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = env.advance(state, direction)
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
        if len(seen) > limit:
            raise ValueError("Level exceeds the validation enumeration budget")
    states = torch.tensor(sorted(seen), dtype=torch.float32, device=env.device)
    return states[~env.success(states, env.goal[:1].expand(len(states), -1))]


def _puzzle_data(env, count, generator):
    states = reachable_states(env)
    if len(states) < 2:
        raise ValueError("Puzzle needs at least two reachable nonterminal states for a disjoint train/holdout split")
    order = torch.randperm(len(states), generator=generator)
    split = max(1, int(0.8 * len(states)))
    pools = states[order[:split]], states[order[split:]]
    data = []
    for pool, n in zip(pools, (count, max(128, count // 4))):
        idx = torch.randint(len(pool), (n,), generator=generator)
        start = pool[idx]
        action = torch.rand(n, env.action_dim, generator=generator) * 2 - 1
        direction, moving = env._directions(action.to(env.device))
        nxt = env._advance_batch(start, direction, moving)
        pixels = torch.stack((env.render(start), env.render(nxt)), 1).cpu()
        actions = torch.stack((action, torch.zeros_like(action)), 1)
        data.append((pixels, actions))
    return data


def _collected_data(env, count, seed):
    datasets = []
    for offset, size in ((0, count), (10000, max(128, count // 4))):
        observations, actions, rewards = [], [], []
        collected, episode = 0, 0
        while collected < size:
            steps = min(32, max(1, (size - collected + env.batch_size - 1) // env.batch_size))
            data = env.collect(steps, generator=torch.Generator().manual_seed(seed + offset + episode))
            valid = data["valid"]
            observations.append(torch.stack((data["observations"][valid], data["next_observations"][valid]), 1).cpu())
            action = data["actions"][valid].cpu()
            actions.append(torch.stack((action, torch.zeros_like(action)), 1))
            rewards.append(data["rewards"][valid].cpu())
            collected += len(action)
            episode += 1
        datasets.append(tuple(torch.cat(parts)[:size] for parts in (observations, actions, rewards)))
    return datasets


def _evaluate_policy(env, policy, seeds, steps, *, goal_scoring):
    scores = {"returns": [], "steps": []}
    if goal_scoring:
        scores["final_distances"] = []
    from torch_world.benchmarks.visualization import capture_frame

    first_video = None
    for seed in seeds:
        torch.manual_seed(seed)
        if hasattr(policy, "reset"):
            policy.reset()
        obs = env.reset(seed=seed)
        total = torch.zeros(env.batch_size, device=env.device)
        lengths = torch.zeros_like(total, dtype=torch.long)
        successes = None
        video = [capture_frame(env, obs)] if first_video is None else None
        for _ in range(min(steps, env.max_episode_steps)):
            active = ~env.transition.done
            if not active.any():
                break
            with torch.no_grad():
                obs = env.step(policy(obs))
            total += env.transition.reward * active
            lengths += active
            if env.transition.success is not None:
                if successes is None:
                    successes = torch.zeros_like(active)
                successes |= env.transition.success & active
            if video is not None:
                video.append(capture_frame(env, obs))
            if env.transition.done.all():
                break
        if goal_scoring:
            successes = env.success(env.state, env.goal)
            scores["final_distances"].extend(env.distance(env.state, env.goal).cpu().tolist())
        scores["returns"].extend(total.cpu().tolist())
        scores["steps"].extend(lengths.cpu().tolist())
        if successes is not None:
            scores.setdefault("successes", []).extend(successes.float().cpu().tolist())
        if first_video is None:
            first_video = torch.stack(video)
    scores["mean_return"] = sum(scores["returns"]) / len(scores["returns"])
    if "successes" in scores:
        scores["success_rate"] = sum(scores["successes"]) / len(scores["successes"])
    return scores, first_video


def observation_encoder(observation_shape):
    """Small 32-dimensional CNN (RGB) or MLP (vectors) for validation experiments."""
    if len(observation_shape) == 3:
        channels, height, width = observation_shape
        encoder = nn.Sequential(
            nn.Conv2d(channels, 16, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * ((height + 3) // 4) * ((width + 3) // 4), 32),
        )
    elif len(observation_shape) == 1:
        encoder = nn.Sequential(nn.Linear(observation_shape[0], 64), nn.GELU(), nn.Linear(64, 32))
    else:
        raise ValueError("Default encoder requires RGB images or observation vectors")
    return encoder


def lewm_baseline(observation_shape, action_dim):
    """Default model factory; replace with any compatible WorldModel factory."""
    encoder = observation_encoder(observation_shape)
    return LeWM(
        encoder,
        action_dim,
        embed_dim=32,
        history=1,
        depth=2,
        num_heads=4,
        head_dim=8,
        mlp_dim=64,
        proj_dim=64,
        dropout=0,
    )


def cem_baseline(action_dim):
    """Default planner factory; replace independently from the model."""
    return CEM(8, action_dim, num_samples=96, num_elites=12, iterations=3, bounds=(-1, 1))


class _RewardObjective:
    def __init__(self, model, head):
        self.model, self.head = model, head

    def rollout(self, *args, **kwargs):
        return self.model.rollout(*args, **kwargs)

    def cost(self, rollout, goal):
        # CEM's goal argument is unused: this objective predicts task rewards.
        return -self.head(rollout[:, -1].flatten(1)).squeeze(-1)


def run_experiment(
    env_id,
    *,
    model_factory=lewm_baseline,
    planner_factory=cem_baseline,
    seed=7,
    updates=400,
    samples=2048,
    evaluation_seeds=4,
    evaluation_steps=60,
    device=None,
    env_kwargs=None,
    output_dir=None,
    save_gifs=True,
    model_config=None,
):
    """Train and compare a swappable WorldModel with a random-action control.

    Pass a world instance, named task class, EnvSpec, or registered string ID.
    Instances are reset during training/evaluation and remain open afterward.
    Classes/recipes are constructed and closed by the runner using env_kwargs.

    ``model_factory(observation_shape, action_dim)`` returns a WorldModel with
    forward(observations, actions)["loss"], encode, predict, rollout and cost.
    This one-step validation protocol requires history=1. Model and planner
    factories are independent of the environment, split, and scoring code.

    Puzzle data splits by starting state; other tasks split by collection seed.
    Goal-renderable tasks use goal MPC, scored by ``success(state, goal)`` when the
    task defines ``distance`` and by its success signal otherwise. Others fit a frozen-latent reward head
    and plan predicted rewards, reporting actual environment returns. Isaac's
    default policy vectors are a state-based experiment, not a pixel result.
    """
    if samples < 2 or updates < 0 or evaluation_seeds < 1 or evaluation_steps < 1:
        raise ValueError("Invalid experiment sample, update or evaluation budget")
    environment = env_id
    spec = get_spec(environment)
    env_id = spec.id
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(seed)
    kwargs = dict(env_kwargs or {})
    if not is_environment(environment):
        if spec.backend == "isaaclab":
            kwargs.setdefault("device", device)
        elif spec.backend == "mujoco":
            kwargs.setdefault("observation", "rgb")
    with environment_scope(environment, **kwargs) as env:
        spec = get_spec(env)
        env_id = spec.id
        tic = time.perf_counter()
        train, test = (
            _puzzle_data(env, samples, generator) if hasattr(env, "advance") else _collected_data(env, samples, seed)
        )
        model = model_factory(tuple(train[0].shape[2:]), env.action_dim).to(device)
        if getattr(model, "history", 1) != 1:
            raise ValueError("This one-step protocol requires a model configured with history=1")
        model_name = type(model).__name__
        planner = planner_factory(env.action_dim)
        planned_name = model_name + " + " + type(planner).__name__
        experiment_config = {
            "protocol": "one-step-world-model-v2",
            "environment": env_id,
            "backend": spec.backend,
            "environment_settings": configuration(env.settings),
            "render_settings": configuration(env.render_settings),
            "model_factory": configuration(model_factory),
            "model_config": configuration(model_config or {}),
            "model_architecture": repr(model),
            "planner_factory": configuration(planner_factory),
            "planner_settings": configuration(vars(planner)),
            "seed": seed,
            "updates": updates,
            "samples": samples,
            "batch_size": 64,
            "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 0.01},
            "evaluation_seeds": [seed + 1000 + i for i in range(evaluation_seeds)],
            "evaluation_steps": evaluation_steps,
            "device": str(device),
            "data_split": "disjoint-start-states" if hasattr(env, "advance") else "disjoint-collection-seeds",
        }
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        losses = []
        model.train()
        for _ in range(updates):
            idx = torch.randint(len(train[0]), (64,), generator=generator)
            output = model(train[0][idx].to(device), train[1][idx].to(device))
            if not torch.isfinite(output["loss"]):
                raise RuntimeError("Nonfinite model loss")
            optimizer.zero_grad()
            output["loss"].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite model gradient")
            optimizer.step()
            losses.append(float(output["loss"].detach()))
        model.eval()
        with torch.no_grad():
            pixels, actions = test[0][:512].to(device), test[1][:512].to(device)
            z = model.encode(pixels)
            prediction = model.predict(z[:, :1], actions[:, :1])
            pred_mse = float((prediction[:, 0] - z[:, 1]).square().mean())
            persistence_mse = float((z[:, 0] - z[:, 1]).square().mean())
            spread = float(z.flatten(0, 1).std(0).mean())
        result = {
            "config": experiment_config,
            "env_id": env_id,
            "model": model_name,
            "observation": "pixels" if train[0].ndim == 5 else "state",
            "planner_policy": planned_name,
            "seed": seed,
            "updates": updates,
            "device": str(device),
            "train_pairs": len(train[0]),
            "test_pairs": len(test[0]),
            "losses": losses,
            "heldout_latent_mse": pred_mse,
            "persistence_latent_mse": persistence_mse,
            "latent_std": spread,
            "torch_version": torch.__version__,
            "evaluation": {},
        }
        env.reset(seed=seed)
        from torch_world.benchmarks.visualization import capture_frame

        result["target"] = capture_frame(env, target=True)
        result["gt_label"] = getattr(env, "gt_label", "GT")
        # A goal observation is planned toward; without `distance`, the simulator's success signal scores it.
        goal_planning = hasattr(env, "goal")
        goal_scoring = goal_planning and hasattr(env, "distance")
        if goal_planning:
            with torch.no_grad():
                target = model.encode(env.render(env.goal).to(device)[:, None])[:, 0]
            objective = model
        else:
            # Same reward head protocol for every user-supplied model.
            reward_head = nn.Sequential(nn.Linear(z[:, 1].flatten(1).shape[-1], 64), nn.GELU(), nn.Linear(64, 1)).to(
                device
            )
            reward_optimizer = torch.optim.AdamW(reward_head.parameters(), lr=1e-3)
            for _ in range(max(updates, 1)):
                idx = torch.randint(len(train[0]), (64,), generator=generator)
                with torch.no_grad():
                    encoded = model.encode(train[0][idx, 1:2].to(device))[:, 0].flatten(1)
                reward_loss = (reward_head(encoded).squeeze(-1) - train[2][idx].to(device)).square().mean()
                reward_optimizer.zero_grad()
                reward_loss.backward()
                reward_optimizer.step()
            reward_head.eval()
            with torch.no_grad():
                result["heldout_reward_mse"] = float(
                    (reward_head(z[:, 1].flatten(1)).squeeze(-1) - test[2][: len(z)].to(device)).square().mean()
                )
            objective = _RewardObjective(model, reward_head)
            # Score every imagined step instead of postponing reward to the horizon.
            if isinstance(planner, CEM):
                planner.objective = "path"
            target = torch.zeros(env.batch_size, device=device)

        def random_policy(obs):
            return torch.rand(env.batch_size, env.action_dim, device=env.device) * 2 - 1

        def planned_policy(obs):
            return planner(objective, model.encode(obs.to(device)[:, None]), target)[:, 0].to(env.device)

        # Record the effective objective, including reward-planning adjustments.
        experiment_config["planner_settings"] = configuration(vars(planner))
        for name, policy in (("random", random_policy), (planned_name, planned_policy)):
            scores, video = _evaluate_policy(
                env,
                policy,
                experiment_config["evaluation_seeds"],
                evaluation_steps,
                goal_scoring=goal_scoring,
            )
            result["evaluation"][name] = scores
            result[name + " frames"] = video
        result["seconds"] = time.perf_counter() - tic
        if output_dir is not None:
            result["run_dir"] = str(save_run(result, output_dir, gifs=save_gifs))
        return result


def show_results(result):
    """Compact notebook report using the library's plots and success intervals."""
    from IPython.display import Markdown, display

    from torch_world import metrics, plots

    model = result.get("model", "LeWM")
    planned = result.get("planner_policy", model + " + CEM")
    display(
        Markdown(
            f"**{result['env_id']} · {model}** — {result.get('observation', 'pixels')} observations, "
            f"seed {result['seed']}, {result['updates']:,} updates, "
            f"{result['train_pairs']:,} training / {result['test_pairs']:,} held-out pairs. "
            f"{result['device']}; {result['seconds']:.1f} s."
        )
    )
    display(
        Markdown(
            "| Prediction method | Held-out latent MSE ↓ |\n|---|---:|\n"
            f"| {model} | {result['heldout_latent_mse']:.4f} |\n"
            f"| Persistence | {result['persistence_latent_mse']:.4f} |\n\n"
            f"Latent standard deviation: **{result['latent_std']:.3f}**. "
            "Latent errors are comparable within this model; their scale differs between models."
        )
    )
    if result["losses"]:
        display(plots.curves({model: result["losses"]}, smooth=25, logy=False, ylabel="training loss"))
    evaluation = result["evaluation"]
    if evaluation:
        goal = "final_distances" in next(iter(evaluation.values()))
        table = [
            "| Policy | Success | Steps | " + ("Final distance ↓" if goal else "Return ↑") + " |",
            "|---|---:|---:|---:|",
        ]
        rates = {}
        for name in (planned, "random"):
            row = evaluation[name]
            successes = row.get("successes")
            if successes is not None:
                rate = metrics.success_rate(torch.tensor(successes).bool())
                rates[name] = rate
                score = str(rate)
            else:
                score = f"{row['success_rate']:.0%}" if "success_rate" in row else "—"
            values = row["final_distances"] if goal else row["returns"]
            table.append(
                f"| {name} | {score} | {sum(row['steps']) / len(row['steps']):.1f} | {sum(values) / len(values):.3f} |"
            )
        display(Markdown("\n".join(table)))
        if rates:
            display(
                plots.bars(
                    {result["env_id"]: rates},
                    xlabel=("final" if goal else "episode") + " success rate (95% Wilson interval)",
                    baseline="random",
                )
            )
    frames = {name: result[name + " frames"] for name in (planned, "random") if name + " frames" in result}
    if frames and next(iter(frames.values())).ndim == 4:
        tiles, titles = [], []
        for video in frames.values():
            indices = torch.linspace(0, len(video) - 1, 6).long()
            tiles.append(torch.cat((result["target"][None], video[indices])))
            titles.extend([result.get("gt_label", "GT"), *(f"t={int(i)}" for i in indices)])
        display(plots.images(torch.cat(tiles), titles=titles, rows=list(frames), ncols=7, size=1.3))
        display(
            Markdown(
                "First evaluation seed shown; both policies start from the same state. All seeds are scored above."
            )
        )
        # Pad ended episodes with their final frame so both videos share a clock.
        length = max(len(video) for video in frames.values())
        videos = torch.stack(
            [torch.cat([video, video[-1:].expand(length - len(video), -1, -1, -1)]) for video in frames.values()]
        )
        target_video = result["target"].expand(length, -1, -1, -1)
        videos = torch.cat((target_video[None], videos))
        titles = [result.get("gt_label", "GT")]
        for name in frames:
            successes = evaluation[name].get("successes")
            label = name if successes is None else f"{'✓ success' if successes[0] else '✗ failure'} · {name}"
            titles.append(label)
        from torch_world.benchmarks.visualization import episode_indices

        indices = episode_indices(length)
        display(
            plots.episodes(videos[:, indices], titles=titles, ncols=3, size=3.0, fps=4, hold=3, frame_steps=indices)
        )
    elif frames:
        from torch_world.benchmarks.visualization import state_figure

        display(state_figure(frames, result["target"]))
        display(Markdown("State observations with dashed GT targets; this is not a camera visualization."))
