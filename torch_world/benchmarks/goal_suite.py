"""Ten goal-reaching tasks from pixels, to compare world models under one protocol.

Every model trains offline on the same episodes of a task for the same number of updates, then plans from 64 x 64
observations toward goal images it has not seen. The tasks are the ones world-model papers report on: PLDM's Two-Rooms,
Push-T (DINO-WM, LeWM), DMControl's Reacher, five OGBench visual tasks with OGBench's own evaluation goals, time limits
and success, and two Meta-World tasks whose goal images come from Meta-World's scripted policies.

    python -m torch_world.benchmarks.goal_suite prepare --task push-t --out runs/goal-suite
    python -m torch_world.benchmarks.goal_suite run --task push-t --model LeWM --seed 0 --out runs/goal-suite

``prepare`` collects or loads a task's training episodes once: they are the same for every model and seed. ``run``
trains one model with one seed, evaluates it and writes ``<out>/<task>/<model>-seed<seed>.json``; ``--model random``
evaluates uniform random actions, the floor. Evaluation goals do not depend on the seed, so seeds measure training
variance.
"""

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from torch_world.planning.mpc import RecedingHorizon

SIZE = 64
UPDATES, BATCH = 2500, 128
MODELS = ("LeWM", "PLDM", "DINO-WM", "V-JEPA 2-AC", "TD-MPC2")


# ---------- evaluation episodes ----------


class Episodes:
    """A batch of evaluation episodes: ``reset`` returns the same first observations ``(B, 3, 64, 64)`` in [0, 1]
    every time, ``goal_images`` are their goals, ``success`` whether each lane is at its goal now. ``valid`` masks
    lanes that are not tasks (a goal the scripted policy never reached)."""

    steps: int
    goal_images: Tensor
    valid: Tensor

    def reset(self) -> Tensor:
        raise NotImplementedError

    def step(self, action: Tensor) -> Tensor:
        return self.env.step(action)

    def success(self) -> Tensor:
        raise NotImplementedError

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()


class StateEpisodes(Episodes):
    """Episodes of an environment that restores, renders and scores any state: start and goal states given."""

    def __init__(self, env, start, goal, steps):
        self.env, self.start, self.goal, self.steps = env, start, goal, steps
        self.goal_images = env.render(goal)
        self.valid = torch.ones(len(start), dtype=torch.bool)

    def reset(self):
        return self.env.reset(self.start)

    def success(self):
        return self.env.success(self.env.state[:, None], self.goal)[:, 0].cpu()


class SeededEpisodes(Episodes):
    """Episodes of a simulator suite, restarted from a seed; success is the suite's, or ``success(env)``."""

    def __init__(self, env, seed, steps, goal_images=None, success=None, valid=None):
        self.env, self.seed, self.steps, self._success = env, seed, steps, success
        self.reset()
        self.goal_images = env.goal.clone() if goal_images is None else goal_images
        self.valid = torch.ones(env.batch_size, dtype=torch.bool) if valid is None else valid.cpu()

    def reset(self):
        return self.env.reset(seed=self.seed)

    def success(self):
        if self._success is not None:
            return self._success(self.env).cpu()
        return self.env.transition.success.cpu()


def run_episodes(episodes: Episodes, policy, frameskip: int) -> dict:
    """Run ``policy(obs) -> (B, action_dim)`` for the budget; returns successes, first-success steps and timing."""
    if hasattr(policy, "reset"):
        policy.reset()
    obs = episodes.reset()
    reached = torch.zeros(len(obs), dtype=torch.bool)
    first = torch.full((len(obs),), -1)
    elapsed = 0.0
    for t in range(episodes.steps):
        tic = time.perf_counter()
        action = policy(obs)
        if action.is_cuda:
            torch.cuda.synchronize(action.device)
        elapsed += time.perf_counter() - tic
        obs = episodes.step(action)
        now = episodes.success()
        first[now & ~reached] = t + 1
        reached |= now
    valid = episodes.valid
    return {
        "success": reached[valid].tolist(),
        "first_success": first[valid].tolist(),
        "ms_per_step": 1000 * elapsed / episodes.steps,
    }


# ---------- tasks ----------


def _uint8(pixels: Tensor) -> Tensor:
    return (pixels.clamp(0, 1) * 255).round().to(torch.uint8).cpu()


def _episodes_from_collect(data: dict) -> dict:
    """``BenchmarkEnv.collect`` output to frames ``(N, T + 1, ...)`` and actions ``(N, T + 1, A)``, the last zero."""
    pixels = torch.cat([data["observations"], data["next_observations"][:, -1:]], dim=1)
    actions = torch.cat([data["actions"], torch.zeros_like(data["actions"][:, :1])], dim=1)
    return {"pixels": _uint8(pixels), "actions": actions.float().cpu()}


@dataclass
class Task:
    name: str
    action_dim: int
    frameskip: int  # environment actions per world-model step
    horizon: int  # planned world-model steps
    steps: int  # environment steps per episode; 0 for the suite's own limit
    source: str
    cached = True  # whether ``prepare`` saves the episodes; OGBench's are read from its dataset file instead

    def data(self) -> dict:
        raise NotImplementedError

    def evaluations(self, device) -> list[Episodes]:
        raise NotImplementedError


class TwoRoomsTask(Task):
    def __init__(self):
        super().__init__("two-rooms", 2, 1, 8, 60, "PLDM (Sobal et al., 2025)")

    def data(self):
        from torch_world.envs import TwoRooms

        data = TwoRooms(batch_size=1024).collect(65, generator=torch.Generator().manual_seed(0))
        return {"pixels": _uint8(data["pixels"]), "actions": data["actions"].float()}

    def evaluations(self, device):
        from torch_world.envs import TwoRooms

        env = TwoRooms(batch_size=64, device=device)
        start, goal = env.tasks(torch.Generator().manual_seed(1))  # the goal in the other room, through the door
        return [StateEpisodes(env, start, goal, self.steps)]


class PushTTask(Task):
    def __init__(self):
        super().__init__("push-t", 2, 5, 5, 50, "DINO-WM, LeWM")

    def data(self):
        from torch_world.envs import PushT

        env = PushT(batch_size=1000, size=SIZE, target=False)
        data = env.collect(100, generator=torch.Generator().manual_seed(0))
        return {"pixels": _uint8(data["pixels"]), "actions": data["actions"].float()}

    def evaluations(self, device):
        from torch_world.envs import PushT

        env = PushT(batch_size=64, size=SIZE, target=False, device=device)
        start, goal = env.tasks(steps=25, generator=torch.Generator().manual_seed(1))
        return [StateEpisodes(env, start, goal, self.steps)]


class ReacherTask(Task):
    """DMControl's reacher-easy from pixels, with action repeat 2. The goal image poses the arm with its fingertip on
    the target (velocities zero); success is the suite's: fingertip and target overlapping."""

    def __init__(self):
        super().__init__("reacher", 2, 5, 5, 100, "DMControl (Tassa et al., 2018)")

    def env(self, batch_size, device=None):
        from torch_world.envs import DMControl

        return DMControl("reacher-easy", batch_size=batch_size, observation="rgb", action_repeat=2, device=device)

    def data(self):
        episodes = [
            _episodes_from_collect(self.env(100).collect(100, generator=torch.Generator().manual_seed(s)))
            for s in range(5)
        ]
        return {k: torch.cat([e[k] for e in episodes]) for k in episodes[0]}

    @staticmethod
    def reached(env):
        out = []
        for lane in range(env.batch_size):
            physics = env.envs[lane].physics
            radii = physics.named.model.geom_size[["target", "finger"], 0].sum()
            out.append(physics.finger_to_target_dist() <= radii)
        return torch.tensor(out)

    @staticmethod
    def goal_images(env):
        """Pose each lane's arm with the fingertip closest to the target over a grid of joint angles, render, restore."""
        frames, grid = [], np.linspace(-np.pi, np.pi, 91)
        for lane in range(env.batch_size):
            physics = env.envs[lane].physics
            saved = physics.get_state().copy()
            best = (np.inf, 0.0, 0.0)
            for shoulder in grid:
                for wrist in grid[(grid > -2.6) & (grid < 2.6)]:  # the wrist joint's range
                    physics.named.data.qpos[["shoulder", "wrist"]] = shoulder, wrist
                    physics.forward()
                    best = min(best, (physics.finger_to_target_dist(), shoulder, wrist))
            physics.named.data.qpos[["shoulder", "wrist"]] = best[1:]
            physics.named.data.qvel[:] = 0
            physics.forward()
            frames.append(env._pixels([lane], env.size))
            physics.set_state(saved)
            physics.forward()
        return torch.cat(frames).to(env.device).float() / 255

    def evaluations(self, device):
        env = self.env(64, device)
        env.reset(seed=10_000)
        goals = self.goal_images(env)
        return [SeededEpisodes(env, 10_000, self.steps, goals, self.reached)]


class OGBenchTask(Task):
    """An OGBench visual task: training episodes from its dataset (the first 200), and its five evaluation goals,
    13 episodes each, with its time limit and success."""

    def __init__(self, name, dataset, action_dim, source="OGBench (Park et al., 2025)"):
        super().__init__(name, action_dim, 5, 5, 0, source)
        self.dataset, self.cached = dataset, False

    def data(self):
        from torch_world.datasets import load_ogbench

        data = load_ogbench(self.dataset)
        return {"pixels": data["observations"][:200].clone(), "actions": data["actions"][:200].clone()}

    def evaluations(self, device):
        from torch_world.envs import OGBench

        out = []
        for task_id in range(1, 6):
            env = OGBench(self.dataset, batch_size=13, task_id=task_id, device=device)
            out.append(SeededEpisodes(env, 1000 * task_id, env.max_episode_steps))
        return out


class MetaWorldTask(Task):
    """A Meta-World v3 task from pixels (the ``corner2`` camera). Training episodes follow Meta-World's scripted
    policy with Gaussian noise (std 0.5), and a fifth are random actions. Evaluation goal images are the frames where
    the scripted policy first succeeds from the same start; lanes where it never does are not tasks."""

    def __init__(self, name, policy):
        super().__init__(name.removesuffix("-v3"), 4, 5, 5, 100, "Meta-World (Yu et al., 2019)")
        self.dataset, self.policy = name, policy

    def env(self, batch_size, device=None, terminate_on_success=True):
        from torch_world.envs import MetaWorld

        return MetaWorld(
            self.dataset,
            batch_size=batch_size,
            observation="rgb",
            size=SIZE,
            terminate_on_success=terminate_on_success,
            device=device,
        )

    def expert(self):
        import metaworld.policies

        return getattr(metaworld.policies, self.policy)()

    def _expert_actions(self, env, policy):
        states = env.state.cpu().numpy()
        return torch.as_tensor(np.stack([policy.get_action(s) for s in states]), dtype=torch.float32).clamp(-1, 1)

    def data(self):
        policy, episodes = self.expert(), []
        for s in range(5):
            env, generator = self.env(100, terminate_on_success=False), torch.Generator().manual_seed(s)
            obs, pixels, actions = env.reset(seed=100 * s), [], []
            random = torch.arange(100) % 5 == 0
            for _ in range(100):
                noise = 0.5 * torch.randn(100, 4, generator=generator)
                action = (self._expert_actions(env, policy) + noise).clamp(-1, 1)
                action[random] = torch.rand(int(random.sum()), 4, generator=generator) * 2 - 1
                pixels.append(obs)
                actions.append(action)
                obs = env.step(action)
            pixels.append(obs)
            actions.append(torch.zeros_like(action))
            episodes.append({"pixels": _uint8(torch.stack(pixels, 1)), "actions": torch.stack(actions, 1)})
            env.close()
        return {k: torch.cat([e[k] for e in episodes]) for k in episodes[0]}

    def evaluations(self, device):
        env, policy, seed = self.env(72, device), self.expert(), 10_000
        obs = env.reset(seed=seed)
        goals, found = obs.clone(), torch.zeros(env.batch_size, dtype=torch.bool, device=obs.device)
        for _ in range(env.max_episode_steps):
            obs = env.step(self._expert_actions(env, policy))
            new = env.transition.success & ~found
            goals[new], found = obs[new], found | new
            if found.all() or env.transition.done.all():
                break
        valid = found & (found.cumsum(0) <= 64)  # the first 64 lanes the scripted policy solves
        return [SeededEpisodes(env, seed, self.steps, goals, valid=valid)]


TASKS = {
    task.name: task
    for task in (
        TwoRoomsTask(),
        PushTTask(),
        ReacherTask(),
        OGBenchTask("antmaze", "visual-antmaze-medium-navigate-v0", 8),
        OGBenchTask("cube", "visual-cube-single-play-v0", 5),
        OGBenchTask("cube-double", "visual-cube-double-play-v0", 5),
        OGBenchTask("scene", "visual-scene-play-v0", 5),
        OGBenchTask("puzzle", "visual-puzzle-3x3-play-v0", 5),
        MetaWorldTask("reach-v3", "SawyerReachV3Policy"),
        MetaWorldTask("push-v3", "SawyerPushV3Policy"),
    )
}


# ---------- training ----------


class Windows:
    """``updates`` random batches of ``length`` frames, every ``frameskip``-th, with the ``frameskip`` actions after
    each frame concatenated (as `TrajectoryDataset`), sampled on the data's device; pixels become floats in [0, 1]."""

    def __init__(self, data, length, frameskip, updates=UPDATES, batch_size=BATCH, generator=None):
        self.data, self.length, self.k, self.updates, self.batch_size = data, length, frameskip, updates, batch_size
        self.generator = generator
        self.num_episodes, self.T = data["actions"].shape[:2]
        if self.T < length * frameskip:
            raise ValueError("Episodes are shorter than one window")

    def sample(self):
        device = self.data["pixels"].device
        e = torch.randint(self.num_episodes, (self.batch_size,), generator=self.generator).to(device)
        t = torch.randint(self.T - self.length * self.k + 1, (self.batch_size,), generator=self.generator).to(device)
        steps = t[:, None] + self.k * torch.arange(self.length, device=device)
        chunks = t[:, None] + torch.arange(self.length * self.k, device=device)
        pixels = self.data["pixels"][e[:, None], steps].float() / 255
        actions = self.data["actions"][e[:, None], chunks].unflatten(1, (self.length, -1)).flatten(2)
        return {"pixels": pixels, "actions": actions}

    def __iter__(self):
        for _ in range(self.updates):
            yield self.sample()


def build(name: str, action_dim: int, steps_per_episode: int):
    """The main benchmark's models: small predictors of equal depth, frozen encoders on 4 x 4 patches."""
    from torch_world.models import DINOWM, PLDM, TDMPC2, LeWM, PLDMEncoder, RoboJEPA
    from torch_world.models.tdmpc import pixel_encoder
    from torch_world.nn import CLSEncoder, PatchTokens, VideoTokens

    predictor = {"depth": 4, "num_heads": 4, "head_dim": 32, "mlp_dim": 512}
    if name == "LeWM":
        encoder = CLSEncoder(img_size=SIZE, patch_size=8, embed_dim=128, depth=4, num_heads=4)
        return LeWM(encoder, action_dim=action_dim, embed_dim=128, proj_dim=512, lamb=0.05, **predictor)
    if name == "PLDM":
        return PLDM(PLDMEncoder(img_size=SIZE, latent_dim=128), action_dim=action_dim, latent_dim=128)
    if name == "DINO-WM":
        encoder = PatchTokens.from_hub("dinov2_vits14", size=56)
        return DINOWM(encoder, num_patches=16, embed_dim=384, action_dim=action_dim, **predictor)
    if name == "V-JEPA 2-AC":
        encoder = VideoTokens.from_hub("vjepa2_1_vit_base_384", size=SIZE)
        return RoboJEPA(embed_dim=768, action_dim=action_dim, encoder=encoder, predictor_dim=256, depth=4, num_heads=4)
    if name == "TD-MPC2":
        # Offline and goal-conditioned, the reference weights (consistency 20, reward and value 0.1) let the
        # consistency loss collapse the encoder to a constant; reward and value carry more weight here.
        discount = TDMPC2.episode_discount(steps_per_episode)
        return TDMPC2(
            pixel_encoder(),
            action_dim=action_dim,
            discount=discount,
            goal_conditioned=True,
            episodic=True,
            consistency_coef=2.0,
            reward_coef=1.0,
            value_coef=1.0,
        )
    raise ValueError(f"Unknown model {name!r}; choose from {MODELS}")


class TemporalGoals(Windows):
    """Goal-conditioned TD-MPC2 batches, relabelled as in offline goal-conditioned RL: the goal is a frame of the same
    episode a geometric number of steps ahead (at least the window, so it never lies inside it), or with probability
    0.3 a frame of another episode. The reward is -1 per step until the goal frame, which ends the episode."""

    def __init__(self, data, frameskip, discount, horizon=3, **kwargs):
        super().__init__(data, horizon + 1, frameskip, **kwargs)
        self.H, self.offset = horizon, torch.distributions.Geometric(probs=torch.tensor(1 - discount))

    def sample(self):
        H, k, B, device = self.H, self.k, self.batch_size, self.data["pixels"].device
        e = torch.randint(self.num_episodes, (B,), generator=self.generator).to(device)
        t = torch.randint(self.T - (H + 1) * k + 1, (B,), generator=self.generator).to(device)
        steps = t[:, None] + k * torch.arange(H + 1, device=device)
        chunks = t[:, None] + torch.arange(H * k, device=device)
        pixels = self.data["pixels"][e[:, None], steps].float() / 255
        actions = self.data["actions"][e[:, None], chunks].unflatten(1, (H, -1)).flatten(2)
        ahead = (H + self.offset.sample((B,))).long().to(device)  # in model steps
        ahead = torch.minimum(ahead, (self.T - 1 - t) // k)
        other = torch.rand(B, generator=self.generator).to(device) < 0.3
        goal_episode = torch.where(
            other, torch.randint(self.num_episodes, (B,), generator=self.generator).to(device), e
        )
        goal_frame = torch.where(other, torch.randint(self.T, (B,), generator=self.generator).to(device), t + k * ahead)
        goal = self.data["pixels"][goal_episode, goal_frame].float() / 255
        reached = ~other[:, None] & (torch.arange(1, H + 1, device=device) == ahead[:, None])
        obs = torch.cat([pixels, goal[:, None].expand_as(pixels)], dim=2)
        return obs, actions, -(~reached).float(), reached.float()


def train(model, name, task, data, generator, updates=UPDATES):
    from torch_world.utils import fit

    k = task.frameskip
    if name == "TD-MPC2":
        batches = TemporalGoals(data, k, model.discount, updates=updates, generator=generator)
        opt = torch.optim.Adam(model.param_groups(3e-4))
        log, model = {}, model.train()
        for obs, actions, rewards, terminated in batches:
            out = model(obs, actions, rewards, terminated)
            opt.zero_grad()
            out["loss"].backward()
            model.clip_grad_norm(20)
            opt.step()
            model.update_target()
            for key, v in out.items():
                log.setdefault(key.removesuffix("_loss"), []).append(v.item())
        model.eval()
        return log
    recipes = {
        "LeWM": {"length": 4},
        "PLDM": {"length": 17, "lr": 1e-3, "weight_decay": 0},
        "DINO-WM": {"length": 4},
        "V-JEPA 2-AC": {"length": 4, "inputs": lambda b: (model.encode(b["pixels"]), b["actions"])},
    }
    recipe = dict(recipes[name])
    return fit(model, Windows(data, recipe.pop("length"), k, updates=updates, generator=generator), epochs=1, **recipe)


# ---------- planning ----------


def policy(model, name, task, goal_images, generator):
    from torch_world import evaluation
    from torch_world.planning import CEM, MPPI

    A = task.frameskip * task.action_dim
    if name == "TD-MPC2":
        mppi = MPPI(horizon=3, action_dim=A, num_samples=256, num_elites=32, num_pi_trajs=12, iterations=4)

        def plan(obs):
            return mppi(model, model.encode(torch.cat([obs, goal_images], dim=1)[:, None]), generator=generator)

        return _MPPIPolicy(plan, task.action_dim, mppi)
    cem = CEM(
        horizon=task.horizon,
        action_dim=A,
        num_samples=128,
        num_elites=16,
        iterations=5,
        bounds=(-1, 1),
        objective="path",
    )
    planner = lambda model, z, goal: cem(model, z, goal, generator=generator)
    if task.frameskip == 1:
        return evaluation.goal_policy(model, planner, goal_images)
    return evaluation.goal_policy(model, planner, goal_images, action_dim=task.action_dim)


class _MPPIPolicy(RecedingHorizon):
    """Receding horizon over MPPI plans; ``reset`` also drops MPPI's warm start between episodes."""

    def __init__(self, plan, action_dim, mppi):
        super().__init__(plan, action_dim)
        self.mppi = mppi

    def reset(self):
        super().reset()
        self.mppi.reset()


def random_policy(task):
    return lambda obs: torch.rand(len(obs), task.action_dim, device=obs.device) * 2 - 1


# ---------- command line ----------


def prepare(task: Task, out: Path):
    path = out / task.name / "data.pt"
    if path.exists() or not task.cached:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tic = time.time()
    data = task.data()
    torch.save(data, path)
    shape = {k: tuple(v.shape) for k, v in data.items()}
    print(f"{task.name}: {shape} in {(time.time() - tic) / 60:.1f} min", flush=True)


def run(task: Task, name: str, seed: int, out: Path, updates: int = UPDATES, bf16: bool = False):
    """Train ``name`` with ``seed`` and evaluate it, or with ``name="random"`` evaluate uniform random actions."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    evaluations = task.evaluations(device)
    result = {"task": task.name, "model": name, "seed": seed, "episodes": {}}
    cuda = torch.Generator(device=device).manual_seed(seed)
    # bf16 autocast, for every model alike, only on request: it can reorder close planning costs
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and device == "cuda")
    if name == "random":
        act = random_policy(task)
    else:
        data = torch.load(out / task.name / "data.pt") if task.cached else task.data()
        data = {k: v.to(device) for k, v in data.items()}
        steps = max(e.steps for e in evaluations)
        model = build(name, task.frameskip * task.action_dim, math.ceil(steps / task.frameskip)).to(device)
        tic = time.time()
        with autocast:
            log = train(model, name, task, data, torch.Generator().manual_seed(seed), updates)
        result["train_minutes"], result["updates"] = (time.time() - tic) / 60, updates
        result["final_loss"] = {k: float(np.mean(v[-50:])) for k, v in log.items()}
        del data
        act = None
    with torch.no_grad(), autocast:
        for i, episodes in enumerate(evaluations):
            plan = act or policy(model, name, task, episodes.goal_images, cuda)
            result["episodes"][i] = run_episodes(episodes, plan, task.frameskip)
            episodes.close()
    successes = [s for r in result["episodes"].values() for s in r["success"]]
    result["success_rate"], result["tasks"] = float(np.mean(successes)), len(successes)
    path = out / task.name / f"{name.replace(' ', '_')}-seed{seed}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"{task.name} {name} seed {seed}: {result['success_rate']:.0%} of {len(successes)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--model", choices=(*MODELS, "random"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/goal-suite"))
    parser.add_argument("--updates", type=int, default=UPDATES, help="training batches (fewer for a smoke test)")
    parser.add_argument("--bf16", action="store_true", help="bf16 autocast in training and planning")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(TASKS[args.task], args.out)
    else:
        run(TASKS[args.task], args.model, args.seed, args.out, args.updates, args.bf16)


if __name__ == "__main__":
    main()
