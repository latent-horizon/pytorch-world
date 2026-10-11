from collections.abc import Callable

import torch
from torch import Tensor


@torch.no_grad()
def run_policy(env, policy: Callable[[Tensor], Tensor], steps: int) -> Tensor:
    """Run ``policy(observation) -> action`` in closed loop from the current state; returns states ``(B, steps + 1, ...)``."""
    obs, states = env.render(), [env.state]
    for _ in range(steps):
        obs = env.step(policy(obs))
        states.append(env.state)
    return torch.stack(states, dim=1)
