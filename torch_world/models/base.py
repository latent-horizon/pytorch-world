import copy

import torch
from torch import Tensor, nn

from torch_world.utils.ema import ema_update


class JEPA(nn.Module):
    """Joint-embedding predictive architecture: predict the embedding of a target from the embedding of a context.

    Subclasses implement ``forward`` returning a dict with a scalar ``"loss"`` and its components.
    """

    def make_target_encoder(self, encoder: nn.Module) -> nn.Module:
        """Frozen copy of ``encoder`` updated by exponential moving average."""
        return copy.deepcopy(encoder).requires_grad_(False)

    def update_target(self, momentum: float):
        ema_update(self.target_encoder, self.encoder, momentum)


class WorldModel(nn.Module):
    """Action-conditioned latent world model.

    Latents are ``(B, T, ...)``. ``predict(z, actions)`` maps ``z[:, t]`` and ``actions[:, t]`` to a prediction of
    ``z[:, t + 1]`` for every ``t`` (teacher forcing). ``rollout`` and ``cost`` build on it for planning.
    """

    history: int | None = None

    def encode(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def predict(self, z: Tensor, actions: Tensor, **cond) -> Tensor:
        raise NotImplementedError

    def rollout(self, z: Tensor, actions: Tensor, **cond) -> Tensor:
        """Unroll from ``T0`` context latents with ``T >= T0`` actions; returns ``T + 1`` latents."""
        for t in range(z.size(1), actions.size(1) + 1):
            s = slice(max(0, t - self.history) if self.history else 0, t)
            z = torch.cat([z, self.predict(z[:, s], actions[:, s], **cond)[:, -1:]], dim=1)
        return z

    def cost(self, z: Tensor, goal: Tensor) -> Tensor:
        """Squared distance between the final latent of each rollout ``(B, T, ...)`` and the goal latent ``(B, ...)``."""
        return (z[:, -1] - goal).square().flatten(1).mean(1)


class WorldActionModel(WorldModel):
    """World model that also *outputs* actions: a policy and a dynamics model in one network.

    ``sample_actions(z, goal)`` draws an action chunk from context latents, ``predict_future(z, actions)`` predicts
    the latents that chunk leads to, and ``imagine(z, goal)`` samples both jointly. ``decode`` maps latents back to
    pixels when the latent space has a decoder, and returns ``None`` otherwise.
    """

    def sample_actions(self, z: Tensor, goal: Tensor | None = None, **kwargs) -> Tensor:
        raise NotImplementedError

    def predict_future(self, z: Tensor, actions: Tensor, goal: Tensor | None = None, **kwargs) -> Tensor:
        raise NotImplementedError

    def imagine(self, z: Tensor, goal: Tensor | None = None, **kwargs) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def decode(self, z: Tensor) -> Tensor | None:
        return None
