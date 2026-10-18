from collections.abc import Callable

import torch
from torch import Tensor

from torch_world.datasets.images import resize_images
from torch_world.datasets.robodojo import pack, robot_layout


class RoboDojoPolicy:
    """Serve a policy to `RoboDojo <https://github.com/RoboDojo-Benchmark/RoboDojo>`_ as an XPolicyLab ``Model``.

    RoboDojo evaluates in Isaac Sim and asks an XPolicyLab policy server for actions over WebSocket, so the policy
    runs in its own environment and is scored on RoboDojo's own layouts. Subclass this as the ``Model`` of an
    XPolicyLab policy folder; the server calls ``reset``, ``update_obs(_batch)`` and ``get_action(_batch)``.

    ``policy`` maps a batch of observations to actions ``(B, A)``, or to chunks ``(B, H, A)`` that RoboDojo executes
    in order. With ``observation="rgb"`` it sees ``camera`` frames ``(B, 3, size, size)`` in [0, 1], resized as
    `load_robodojo` resizes them for training; with ``"state"``, the packed robot vector ``(B, D)`` of
    `robot_layout`. Actions use the same packing. ``action_bounds=(low, high)``, each ``(A,)``, maps policy actions
    in [-1, 1] to robot units, for example the range of the training actions; without it they are robot units.
    With ``relative=True`` the policy's joint actions are offsets from the current robot state, and the robot is sent
    ``state + action``; at 25 Hz consecutive joint targets are nearly the current state, so offsets are the
    informative quantity to learn and plan.
    """

    def __init__(
        self,
        policy: Callable[[Tensor], Tensor],
        *,
        observation: str = "rgb",
        camera: str = "cam_head",
        size: int = 64,
        action_type: str = "joint",
        robot: dict | None = None,
        action_bounds: tuple | None = None,
        relative: bool = False,
        device: str | torch.device | None = None,
    ):
        if observation not in ("rgb", "state"):
            raise ValueError("observation must be 'rgb' or 'state'")
        if relative and action_type != "joint":
            raise ValueError("relative actions are joint offsets; poses do not add")
        self.policy, self.observation, self.camera, self.size = policy, observation, camera, size
        self.relative = relative
        self.layout = robot_layout(action_type, robot)
        self.action_dim = sum(width for _, width in self.layout)
        self.device = torch.device(device or "cpu")
        self.bounds = None
        if action_bounds is not None:
            self.bounds = tuple(torch.as_tensor(b, dtype=torch.float32, device=self.device) for b in action_bounds)
            if any(b.shape != (self.action_dim,) for b in self.bounds):
                raise ValueError(f"action_bounds must be two tensors of shape ({self.action_dim},)")
        self.reset()

    def observe(self, obs: dict) -> Tensor:
        """One RoboDojo observation dict as the policy's input, without the batch dimension."""
        if self.observation == "state":
            return torch.from_numpy(pack(obs["state"], self.layout))
        return resize_images(obs["vision"][self.camera]["color"], self.size).float() / 255

    def update_obs(self, obs: dict):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list: list[dict]):
        self._latest = [obs.get("env_idx", 0) for obs in obs_list]
        self._observations.update(zip(self._latest, map(self.observe, obs_list)))
        if self.relative:
            self._states.update(
                (i, torch.from_numpy(pack(obs["state"], self.layout))) for i, obs in zip(self._latest, obs_list)
            )

    def get_action(self) -> list[dict]:
        if not self._latest:
            raise RuntimeError("get_action() called before update_obs()")
        return self.get_action_batch(self._latest[:1])[0]

    def get_action_batch(self, env_idx_list: list[int] | None = None) -> list[list[dict]]:
        """An action chunk per environment, each action a dict of robot fields in robot units."""
        env_idx_list = self._latest if env_idx_list is None else env_idx_list
        if not env_idx_list:
            raise RuntimeError("get_action_batch() called before update_obs_batch()")
        observations = torch.stack([self._observations[i] for i in env_idx_list]).to(self.device)
        with torch.no_grad():
            actions = torch.as_tensor(self.policy(observations), dtype=torch.float32, device=self.device)
        actions = actions[:, None] if actions.ndim == 2 else actions
        expected = actions.ndim == 3 and actions.shape[::2] == (len(env_idx_list), self.action_dim)
        if not expected or not torch.isfinite(actions).all():
            raise ValueError(f"policy must return finite actions (B, {self.action_dim}) or chunks (B, H, A)")
        if self.bounds is not None:
            low, high = self.bounds
            actions = low + (actions.clamp(-1, 1) + 1) / 2 * (high - low)
        if self.relative:
            actions = actions + torch.stack([self._states[i] for i in env_idx_list]).to(self.device)[:, None]
        return [[self._unpack(action) for action in chunk] for chunk in actions.cpu().numpy()]

    def _unpack(self, action) -> dict:
        fields, offset = {}, 0
        for key, width in self.layout:
            fields[key], offset = action[offset : offset + width], offset + width
        return fields

    def reset(self):
        self._observations, self._states, self._latest = {}, {}, []
        if hasattr(self.policy, "reset"):
            self.policy.reset()
