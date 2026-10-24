import torch
from torch import Tensor, nn

from torch_world.models.base import WorldModel
from torch_world.models.jepa._lewm import LeWM
from torch_world.nn.transformer import MLP, Transformer


class MacroLevel(WorldModel):
    """Upper level of a hierarchical world model: predicts the next waypoint latent from past waypoint latents and
    latent macro-actions, with a causal transformer conditioned on the macro-actions through adaLN-zero.

    Macro-actions are standardized: the encoders that produce them end with a BatchNorm without affine parameters, so
    each dimension of a training macro-action has zero mean and unit variance. Planners search them in ``N(0, I)``
    clipped to about ``±2``, as the papers initialize from and clip to the statistics of training macro-actions.
    ``p`` is the norm of the training loss and the planning cost: 1 in HWM, 2 in H-JEPA.
    """

    def __init__(
        self,
        embed_dim: int,
        macro_dim: int,
        history: int = 3,
        depth: int = 4,
        num_heads: int = 8,
        head_dim: int = 24,
        mlp_dim: int = 1024,
        dropout: float = 0.1,
        p: int = 2,
    ):
        super().__init__()
        self.history, self.macro_dim, self.p = history, macro_dim, p
        self.macro_embed = MLP([macro_dim, 4 * embed_dim, embed_dim], act=nn.SiLU)
        self.pos_embed = nn.Parameter(torch.randn(1, history, embed_dim))
        self.predictor = Transformer(
            embed_dim, depth, num_heads, conditional=True, mlp_dim=mlp_dim, head_dim=head_dim, dropout=dropout
        )
        self.proj = nn.Linear(embed_dim, embed_dim)

    def predict(self, z: Tensor, macro: Tensor) -> Tensor:
        """Waypoint latents ``(B, T, embed_dim)`` and macro-actions ``(B, T, macro_dim)`` to next waypoint latents."""
        h = self.predictor(z + self.pos_embed[:, : z.size(1)], self.macro_embed(macro), is_causal=True)
        return self.proj(h)

    def distance(self, pred: Tensor, target: Tensor) -> Tensor:
        """Mean ``p``-distance over the last dim: absolute error for ``p = 1``, squared error for ``p = 2``."""
        return (pred - target).abs().pow(self.p).mean(-1)

    def cost(self, z: Tensor, goal: Tensor) -> Tensor:
        return self.distance(z[:, -1], goal)


class HWM(WorldModel):
    """Hierarchical world model (HWM): two action-conditioned predictors in one latent space, on two time scales.

    The low level is a ``LeWM`` that predicts the next latent from primitive actions. The high level predicts the
    latent ``stride`` steps later, the next waypoint, from a latent macro-action: an MLP encodes the ``stride``
    actions between two waypoints. Both levels share the low level's latents; the high level sees them through a
    stop-gradient, as HWM trains it on a frozen encoder. Planning is top-down: a planner searches macro-actions that
    reach the goal latent, the first predicted waypoint becomes the subgoal of a low-level planner, and the plan is
    redone after each executed step (``planning.HierarchicalPlanner``).

    Windows hold ``high.history + 1`` waypoints, ``T = high.history * stride + 1`` frames. Each level is trained with
    the ℓ1 teacher-forcing loss, the low level also with the ℓ1 multi-step rollout loss on a random crop of
    ``low.history + 1`` frames, and SIGReg keeps the latents from collapsing as the encoder learns from pixels::

        loss = tf_low + gamma_roll * roll_low + lamb * sigreg + tf_high

    Zhang et al., "Hierarchical Planning with Latent World Models", 2026.
    Paper: https://arxiv.org/abs/2604.03208
    Code: https://github.com/kevinghst/HWM_PLDM

    Args:
        low: the low-level world model; its ``lamb`` and ``sigreg`` weight SIGReg.
        action_dim: dimension of one primitive action.
        stride: primitive steps between waypoints.
        macro_dim: dimension of a macro-action (8 in the paper's maze).
    """

    def __init__(
        self,
        low: LeWM,
        action_dim: int,
        stride: int = 4,
        macro_dim: int = 8,
        embed_dim: int = 192,
        history: int = 3,
        depth: int = 4,
        num_heads: int = 8,
        head_dim: int = 24,
        mlp_dim: int = 1024,
        dropout: float = 0.1,
        gamma_roll: float = 1.0,
    ):
        super().__init__()
        self.low, self.stride, self.gamma_roll = low, stride, gamma_roll
        self.history = low.history
        self.macro_encoder = nn.Sequential(
            MLP([stride * action_dim, 4 * embed_dim, macro_dim]), nn.BatchNorm1d(macro_dim, affine=False)
        )
        self.high = MacroLevel(embed_dim, macro_dim, history, depth, num_heads, head_dim, mlp_dim, dropout, p=1)

    def encode(self, pixels: Tensor) -> Tensor:
        """``(B, T, C, H, W)`` to ``(B, T, embed_dim)``."""
        return self.low.encode(pixels)

    def predict(self, z: Tensor, actions: Tensor) -> Tensor:
        return self.low.predict(z, actions)

    def cost(self, z: Tensor, goal: Tensor) -> Tensor:
        return self.high.cost(z, goal)

    def lift(self, z: Tensor) -> Tensor:
        """Low-level latents to high-level latents: the identity, as both levels share one latent space."""
        return z

    def encode_macro(self, actions: Tensor) -> Tensor:
        """Primitive actions ``(B, K * stride, A)`` to macro-actions ``(B, K, macro_dim)``."""
        chunks = actions.unflatten(1, (-1, self.stride)).flatten(2)
        return self.macro_encoder(chunks.flatten(0, 1)).unflatten(0, chunks.shape[:2])

    def forward(self, pixels: Tensor, actions: Tensor) -> dict[str, Tensor]:
        """``pixels`` ``(B, T, C, H, W)`` and ``actions`` ``(B, T, A)`` with ``T = high.history * stride + 1``."""
        z = self.encode(pixels)
        h = self.low.history
        t = int(torch.randint(z.size(1) - h, ()))
        pred = self.low.predict(z[:, t : t + h], actions[:, t : t + h])
        tf_low = (pred - z[:, t + 1 : t + h + 1]).abs().mean()
        roll = self.low.rollout(z[:, t : t + 1], actions[:, t : t + h])[:, 1:]
        roll_low = (roll - z[:, t + 1 : t + h + 1]).abs().mean()
        sigreg = self.low.sigreg(z.transpose(0, 1))
        waypoints = z.detach()[:, :: self.stride]
        macro = self.encode_macro(actions[:, : (waypoints.size(1) - 1) * self.stride])
        tf_high = self.high.distance(self.high.predict(waypoints[:, :-1], macro), waypoints[:, 1:]).mean()
        loss = tf_low + self.gamma_roll * roll_low + self.low.lamb * sigreg + tf_high
        return {"loss": loss, "low_loss": tf_low, "rollout_loss": roll_low, "sigreg_loss": sigreg, "high_loss": tf_high}
