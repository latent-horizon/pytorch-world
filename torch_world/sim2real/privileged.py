"""Simulation-only physical-state baseline for diagnosing visual LeWAM control.

Explicit inputs contain no expert phase, counter or future action. This adapter
rejects ordinary observations and is not supported by hardware checkpoint loading.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from torch_world.models import LeWAM
from torch_world.planning import LeWAMPlanner
from torch_world.sim2real.control import Observation
from torch_world.sim2real.models import ModelAdapter, WorldModelPolicy


@dataclass
class OracleObservation(Observation):
    """Deliberately separate simulation diagnostic input; never a hardware input."""

    privileged: torch.Tensor | None = None


def oracle_observation(env, observation, *, goal=False):
    """28 physical coordinates, with no expert phase, counter or future actions."""
    tool = env.controller.pose(env.reference_joints) if goal else env.tool_position.copy()
    piece = env.target_position.copy() if goal else env.position.copy()
    up = np.array([0.0, 0.0, 1.0]) if goal else env.data.xmat[env.piece_body].reshape(3, 3)[:, 2].copy()
    dof = env.model.joint("piece").dofadr[0]
    velocity = np.zeros(6) if goal else env.data.qvel[dof : dof + 6].copy()
    state = torch.cat(
        (
            observation.state(True),
            torch.tensor(np.r_[tool, piece, env.target_position, up, velocity], dtype=torch.float32),
        )
    )
    return OracleObservation(
        observation.rgb,
        observation.joints,
        observation.opening,
        observation.timestamp,
        observation.contract,
        observation.commanded,
        state,
    )


class ZeroEncoder(nn.Module):
    """No visual information or visual optimization in the oracle baseline."""

    def forward(self, pixels):
        return pixels.new_zeros(len(pixels), 32)


class OracleAdapter(ModelAdapter):
    def proprio(self, state):
        if state.shape[-1] != 28 or not torch.isfinite(state).all():
            raise ValueError("Oracle requires 28 finite simulation-state coordinates")
        scale = state.new_tensor([torch.pi] * 5 + [1, 0.5, 0.5, 0.5, 1] + [0.5] * 9 + [1] * 9)
        return state / scale

    def observation_state(self, observation):
        if not isinstance(observation, OracleObservation) or observation.privileged is None:
            raise ValueError("Oracle policy requires simulation-only OracleObservation")
        return observation.privileged


def make_oracle(seed):
    torch.manual_seed(seed)
    model = LeWAM(
        ZeroEncoder(),
        4,
        embed_dim=32,
        history=16,
        chunk_size=8,
        dim=64,
        depth=2,
        num_heads=4,
        mlp_dim=128,
        state_dim=28,
        action_scale=(1, 1, 1, 0.15),
        lamb=0,
    )
    # Freeze the constant visual branch; SIGReg is irrelevant to exact state.
    for parameter in model.projector.parameters():
        parameter.requires_grad_(False)
    return OracleAdapter(
        model,
        {
            "kind": "lewam",
            "history": 16,
            "chunk_size": 8,
            "proprioception": True,
            "encoder": "simulation_oracle",
            "simulation_only": True,
        },
    )


class OraclePolicy:
    def __init__(self, adapter, env, goal, *, seed, noise):
        self.env = env
        self.policy = WorldModelPolicy(
            adapter, oracle_observation(env, goal, goal=True), seed=seed, planner=LeWAMPlanner(noise_scale=noise)
        )
        self.latencies = self.policy.latencies

    def __call__(self, observation):
        return self.policy(oracle_observation(self.env, observation))

    def record_action(self, action):
        self.policy.record_action(action)
