"""Deterministic reference actions, replayed and verified by ``show_success``.

References demonstrate the objective; they are never model evaluation results.
Contributors may override ``BenchmarkEnv.reference_policy`` with a controller that
reads the world state, or supply a policy directly to ``show_success``.
"""

import json
from collections import deque
from importlib.resources import files

import torch


class ActionSequence:
    """Replay normalized actions on the world's device, broadcasting to its lanes."""

    def __init__(self, world, actions):
        self.actions = torch.as_tensor(actions, dtype=torch.float32, device=world.device)
        if self.actions.ndim != 2 or self.actions.shape[1] != world.action_dim or not len(self.actions):
            raise ValueError("Reference actions must have shape (steps, action_dim)")
        if not torch.isfinite(self.actions).all() or self.actions.abs().max() > 1:
            raise ValueError("Reference actions must be finite and normalized to [-1, 1]")
        self.batch_size = world.batch_size
        self.index = 0
        self.max_steps = len(self.actions)

    def __call__(self, observation):
        if self.index >= len(self.actions):
            raise ValueError("Reference actions exhausted before a verified success")
        action = self.actions[self.index].expand(self.batch_size, -1)
        self.index += 1
        return action


class ParkingReference:
    """Correct heading drift while driving the differential robot toward its bay."""

    def __init__(self, world, max_steps):
        self.world, self.max_steps = world, max_steps

    def __call__(self, observation):
        state, goal = self.world.state, self.world.goal
        speed = (20 * (goal[:, 0] - state[:, 0])).clamp(-1, 1)
        turn = (
            4 * (state[:, 2] - goal[:, 2]) + 4 * (state[:, 1] - goal[:, 1]) + state[:, self.world.model.nq + 2]
        ).clamp(-0.4, 0.4)
        return torch.stack((speed + turn, speed - turn), -1).clamp(-1, 1)


def puzzle_actions(world):
    """Find a shortest successful path through the puzzle's exact transition rules."""
    queue = deque([world.initial])
    parents = {world.initial: None}
    while queue:
        state = queue.popleft()
        if world.success(torch.tensor([state], device=world.device), world.goal[:1]).item():
            path = []
            while parents[state] is not None:
                state, action = parents[state]
                path.append(action)
            return list(reversed(path)) or [(0, 0)]
        for action in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_state = world.advance(state, action)
            if next_state not in parents:
                parents[next_state] = (state, action)
                queue.append(next_state)
    raise ValueError("No successful path exists in this puzzle")


def reference_policy(world):
    """Load a built-in plan; success must still be checked in the current simulator."""
    from torch_world.envs.registry import get_spec

    name = get_spec(world).id
    path = files("torch_world.envs").joinpath("reference_actions", f"{name}.json")
    if not path.is_file():
        raise ValueError(f"{name}: no built-in successful episode is available; pass policy=your_controller")
    reference = json.loads(path.read_text())
    if reference.get("controller") == "parking":
        return ParkingReference(world, reference["max_steps"])
    return ActionSequence(world, reference["actions"])
