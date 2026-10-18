import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor

from torch_world import metrics


@dataclass
class GoalReaching:
    """Closed-loop episodes of each policy from the same starts toward the same goals.

    Per policy: ``paths`` are states ``(N, steps + 1, state_dim)``, ``distances`` the distance to the goal at every
    step ``(N, steps + 1)``, ``reached`` whether each step counts as reaching it ``(N, steps + 1)``, ``success``
    whether the final step does ``(N,)``, ``actions`` the actions taken ``(N, steps, action_dim)`` and ``seconds``
    the mean wall time of one policy call. ``positions`` ``(N, steps + 1, D)`` and ``goals`` ``(N, D)`` locate what the
    task moves (``env.object_position``), for SPL, ``coverage`` ``(N, steps + 1)`` is the goal-region overlap for environments that
    define it (Push-T), and ``frames`` holds uint8 videos ``(N, steps + 1, 3, H, W)`` when ``record`` was given.
    """

    paths: dict[str, Tensor]
    distances: dict[str, Tensor]
    success: dict[str, Tensor]
    reached: dict[str, Tensor] = field(default_factory=dict)
    actions: dict[str, Tensor] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)
    positions: dict[str, Tensor] = field(default_factory=dict)
    goals: Tensor | None = None
    coverage: dict[str, Tensor] = field(default_factory=dict)
    frames: dict[str, Tensor] = field(default_factory=dict)

    def rates(self) -> dict[str, float]:
        """Final success rate of each policy."""
        return {name: s.float().mean().item() for name, s in self.success.items()}

    def intervals(self, confidence: float = 0.95) -> dict[str, metrics.Rate]:
        """Final success rate of each policy with its Wilson interval, as ``plots.bars`` draws it."""
        return {name: metrics.success_rate(s, confidence) for name, s in self.success.items()}

    def summary(self) -> dict[str, dict[str, float | metrics.Rate]]:
        """The standard measures of goal-reaching episodes, per policy:

        - ``success``: final success rate with its 95% Wilson interval;
        - ``reached``: fraction of episodes that reach the goal at any step;
        - ``steps``: mean number of steps to first reach the goal, over the episodes that do;
        - ``final distance``: median distance to the goal at the end;
        - ``progress``: median fraction of the initial distance closed, over episodes that do not start at the goal;
        - ``SPL``: success weighted by path length of the scored object;
        - ``smoothness``: mean change between consecutive actions (lower is smoother);
        - ``ms / step``: wall time of one policy call;
        - ``coverage``: mean of the best goal-region overlap of each episode, when the environment defines it.
        """
        out = {}
        for name, success in self.success.items():
            row = {"success": metrics.success_rate(success)}
            if name in self.reached:
                first = metrics.first_success(self.reached[name])
                row["reached"] = (first >= 0).float().mean().item()
                row["steps"] = first[first >= 0].float().mean().item() if (first >= 0).any() else float("nan")
            row["final distance"] = self.distances[name][:, -1].median().item()
            row["progress"] = metrics.progress(self.distances[name]).nanmedian().item()
            if name in self.positions and self.goals is not None:
                row["SPL"] = metrics.spl(success, self.positions[name], self.goals)
            if name in self.actions and self.actions[name].size(1) > 1:
                row["smoothness"] = metrics.smoothness(self.actions[name]).mean().item()
            if name in self.seconds:
                row["ms / step"] = 1000 * self.seconds[name]
            if name in self.coverage:
                row["coverage"] = self.coverage[name].max(1).values.mean().item()
            out[name] = row
        return out

    def __repr__(self) -> str:
        summary = self.summary()
        columns = list(dict.fromkeys(k for row in summary.values() for k in row))
        width = max(map(len, summary)) + 2
        lines = [
            " " * width + "".join(f"{c:>17s}" if c == "success" else f"{c:>{max(len(c), 6) + 2}s}" for c in columns)
        ]
        for name, row in summary.items():
            cells = []
            for c in columns:
                w = 17 if c == "success" else max(len(c), 6) + 2
                v = row.get(c, float("nan"))
                text = (
                    str(v)
                    if isinstance(v, metrics.Rate)
                    else f"{v:.0%}"
                    if c in ("reached", "coverage")
                    else f"{v:.3g}"
                )
                cells.append(f"{text:>{w}s}")
            lines.append(f"{name:{width}s}" + "".join(cells))
        return "\n".join(lines)


@torch.no_grad()
def goal_reaching(
    env,
    policies: dict[str, Callable[[Tensor], Tensor]],
    start: Tensor,
    goal: Tensor,
    steps: int,
    record: Callable[[object], Tensor] | None = None,
) -> GoalReaching:
    """Run each ``policy(observation) -> action`` for ``steps`` from ``start`` and score it against ``goal`` states.

    The environment provides ``distance(states, goal)`` and ``success(states, goal)``, and optionally
    ``object_position(states)`` (for SPL, in environments where a path length is meaningful) and
    ``coverage(states, goal)``. ``record(env)``, if given, returns images
    ``(N, 3, H, W)`` of the batch at every step, for example ``lambda env: env.render(size=192)``.
    """
    paths, distances, success, frames = {}, {}, {}, {}
    reached, actions, seconds, positions, coverage = {}, {}, {}, {}, {}
    for name, policy in policies.items():
        obs = env.reset(start)
        states, video, taken, elapsed = [env.state], [], [], 0.0
        for t in range(steps + 1):
            if record is not None:
                video.append((record(env).clamp(0, 1) * 255).round().to(torch.uint8).cpu())
            if t < steps:
                tic = time.perf_counter()
                action = policy(obs)
                elapsed += time.perf_counter() - tic
                taken.append(torch.as_tensor(action).float().cpu())
                obs = env.step(action)
                states.append(env.state)
        paths[name] = torch.stack(states, dim=1)
        distances[name] = env.distance(paths[name], goal)
        reached[name] = env.success(paths[name], goal)
        success[name] = reached[name][:, -1]
        actions[name] = torch.stack(taken, dim=1)
        seconds[name] = elapsed / max(steps, 1)
        if hasattr(env, "object_position"):
            positions[name] = env.object_position(paths[name])
        if hasattr(env, "coverage"):
            coverage[name] = env.coverage(paths[name], goal)
        if record is not None:
            frames[name] = torch.stack(video, dim=1)
    goals = env.object_position(goal) if hasattr(env, "object_position") else None
    return GoalReaching(paths, distances, success, reached, actions, seconds, positions, goals, coverage, frames)


def random_policy(env) -> Callable[[Tensor], Tensor]:
    """Uniform actions in ``[-1, 1]^action_dim``."""
    return lambda obs: torch.rand(len(obs), env.action_dim, device=obs.device) * 2 - 1
