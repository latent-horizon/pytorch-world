from collections.abc import Callable

from torch import Tensor


class RecedingHorizon:
    """Closed-loop policy from a planner: plan, run the first ``execute`` planned steps, then plan again.

    ``plan(observation)`` returns plans ``(B, horizon, k * action_dim)`` whose steps are chunks of ``k``
    environment actions, as models trained with a frameskip of ``k`` predict one latent per chunk. Each call
    returns one environment action ``(B, action_dim)``. DINO-WM and LeWM execute the whole plan (``execute`` equal
    to the horizon) before planning again; ``execute=1`` replans after every chunk. Call ``reset`` between episodes.
    """

    def __init__(self, plan: Callable[[Tensor], Tensor], action_dim: int, execute: int = 1):
        self.plan, self.action_dim, self.execute = plan, action_dim, execute
        self.queue: list[Tensor] = []
        self.calls = 0

    def reset(self):
        self.queue = []

    def __call__(self, observation: Tensor) -> Tensor:
        if not self.queue:
            plan = self.plan(observation)[:, : self.execute]
            self.queue = list(plan.reshape(len(plan), -1, self.action_dim).unbind(1))
            self.calls += 1
        return self.queue.pop(0)
