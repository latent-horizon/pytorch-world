"""Seed-matched evaluation for episodic tasks, including non-goal control."""

from collections.abc import Callable, Sequence

import torch


@torch.no_grad()
def benchmark(env, policies: dict[str, Callable], seeds: Sequence[int] = (0, 1, 2), steps: int | None = None):
    """Compare policies on identical seeded episodes, respecting termination.

    Returns CPU tensors per policy: returns, lengths, terminated, truncated and
    optionally success (ever achieved). Missing success stays absent. Policies
    may implement ``reset()`` to clear recurrent/planning state between seeds.
    A seed creates ``env.batch_size`` episodes. Evaluation cutoff is truncation.
    """
    seeds = tuple(seeds)
    steps = env.max_episode_steps if steps is None else steps
    if not seeds or steps < 1:
        raise ValueError("Provide at least one seed and a positive step budget")
    result = {}
    for name, policy in policies.items():
        batches = []
        for seed in seeds:
            torch.manual_seed(seed)
            if hasattr(policy, "reset"):
                policy.reset()
            obs = env.reset(seed=seed)
            returns = torch.zeros(env.batch_size, device=env.device)
            lengths = torch.zeros(env.batch_size, dtype=torch.long, device=env.device)
            succeeded = None
            for _ in range(steps):
                active = ~env.transition.done
                if not active.any():
                    break
                obs = env.step(policy(obs))
                outcome = env.transition
                returns += outcome.reward * active
                lengths += active
                if outcome.success is not None:
                    if succeeded is None:
                        succeeded = torch.zeros_like(active)
                    succeeded |= outcome.success & active
            row = {
                "returns": returns,
                "lengths": lengths,
                "terminated": env.transition.terminated,
                "truncated": env.transition.truncated | ~env.transition.done,
            }
            if succeeded is not None:
                row["success"] = succeeded
            batches.append(row)
        result[name] = {k: torch.cat([b[k].cpu() for b in batches]) for k in batches[0]}
    return result
