from collections.abc import Callable

import torch
from torch import Tensor

from torch_world.models.base import WorldModel
from torch_world.safety.reachability import SafetyValue, imagine_step


class SafetyFilter:
    """Least-restrictive latent safety filter: keep the task action unless it leads where the safety value is low.

    The world model imagines the next latent under the task's action; if the safety value there stays above
    ``threshold`` the task action runs, otherwise the safety policy ``pi(z)`` of the value function takes over:

    ``a = a_task  if V(f(z, a_task)) > threshold  else  pi(z)``

    A ``threshold`` above zero absorbs errors of the learned value and of the predicted latent; the paper uses 0.4 in
    simulation and 0.3 on hardware. ``rule="q"`` checks ``Q(z, a_task) > threshold`` instead, which needs no
    world-model call; the critic was trained on the same imagined transitions, so the two rules agree when it fits.

    Nakamura, Peters and Bajcsy, "Generalizing Safety Beyond Collision-Avoidance via Latent-Space Reachability Analysis", 2025.
    Paper: https://arxiv.org/abs/2502.00935
    Code: https://github.com/CMU-IntentLab/latent-safety
    """

    def __init__(self, model: WorldModel, value: SafetyValue, threshold: float = 0.4, rule: str = "value"):
        if rule not in ("value", "q"):
            raise ValueError(f"rule must be 'value' or 'q', not {rule!r}")
        self.model, self.value, self.threshold, self.rule = model, value, threshold, rule

    @torch.no_grad()
    def safe(self, z: Tensor, actions: Tensor) -> Tensor:
        """Whether each task action ``(B, action_dim)`` keeps latents ``(B, *latent_shape)`` safe ``(B,)``."""
        if self.rule == "q":
            return self.value.q(z, actions) > self.threshold
        return self.value.value(imagine_step(self.model, z, actions)) > self.threshold

    @torch.no_grad()
    def __call__(self, z: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        """Filtered actions ``(B, action_dim)`` and whether the safety policy overrode each ``(B,)``."""
        safe = self.safe(z, actions)
        return torch.where(safe[:, None], actions, self.value.action(z).to(actions)), ~safe

    def wrap(self, policy: Callable[[Tensor], Tensor], **encode) -> "FilteredPolicy":
        """Closed-loop policy ``observation -> action`` that filters ``policy``; see ``FilteredPolicy``."""
        return FilteredPolicy(self, policy, encode)


class FilteredPolicy:
    """A task policy ``observation -> action`` behind a ``SafetyFilter``.

    Observations ``(B, C, H, W)`` are encoded with ``model.encode`` (one frame, on the model's device, ``encode``
    holding extra keyword arguments). ``overrides`` records, per call, which episodes the safety policy took over
    ``(B,)``; ``reset`` clears it, and resets the task policy when it has a ``reset``.
    """

    def __init__(self, safety_filter: SafetyFilter, policy: Callable[[Tensor], Tensor], encode: dict):
        self.filter, self.policy, self.encode = safety_filter, policy, encode
        self.device = next(safety_filter.model.parameters()).device
        self.overrides: list[Tensor] = []

    def reset(self):
        self.overrides = []
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    @torch.no_grad()
    def __call__(self, observation: Tensor) -> Tensor:
        action = torch.as_tensor(self.policy(observation))
        z = self.filter.model.encode(observation[:, None].to(self.device), **self.encode)[:, 0]
        filtered, override = self.filter(z, action.to(self.device).float())
        self.overrides.append(override.cpu())
        return filtered.to(action.device)
