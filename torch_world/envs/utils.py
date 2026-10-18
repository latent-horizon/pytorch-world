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


class NumpyStreams:
    """One stream of NumPy's global generator per batch lane, for simulators that draw from ``np.random``.

    ``call`` runs an upstream function with its lane's stream as the global generator and then restores the caller's,
    so seeded resets are reproducible and the caller's own draws are untouched.
    """

    def __init__(self, lanes: int):
        import numpy as np

        self.np = np
        self.states = [np.random.get_state() for _ in range(lanes)]

    def seed(self, lane: int, seed: int):
        self.states[lane] = self.np.random.RandomState(seed).get_state()

    def call(self, lane: int, fn, *args, **kwargs):
        outer = self.np.random.get_state()
        self.np.random.set_state(self.states[lane])
        try:
            return fn(*args, **kwargs)
        finally:
            self.states[lane] = self.np.random.get_state()
            self.np.random.set_state(outer)
