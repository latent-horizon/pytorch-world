import torch
from torch import Tensor, nn

from torch_world.losses.sigreg import SIGReg
from torch_world.models.base import WorldModel
from torch_world.nn.transformer import MLP, Transformer


class GradientReversal(torch.autograd.Function):
    """Identity forward; multiplies the gradient by ``-scale`` backward (Ganin & Lempitsky, 2015)."""

    @staticmethod
    def forward(ctx, x: Tensor, scale: float) -> Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: Tensor):
        return -ctx.scale * grad, None


class CFJEPA(WorldModel):
    """CF-JEPA: a LeWM-style JEPA world model whose latent is factored into a controllable and an uncontrollable part,
    so that distractors which move on their own stay out of the part the planner looks at.

    Each frame is encoded to one embedding (the encoder's CLS token) and projected, as in LeWM, then split into
    ``z = [z^c, z^u]`` (``ctrl_dim`` and ``embed_dim - ctrl_dim`` dims). A causal transformer ``g``, conditioned on the
    actions through adaLN-zero, predicts the next ``z^c`` from ``z^c`` alone; a smaller, action-free causal transformer
    ``h`` predicts the next ``z^u`` from ``z^u`` alone. An inverse model ``psi([z^c_t, z^c_{t+1}])`` must recover the
    action, so ``z^c`` holds what the agent controls; an adversary ``phi([z^u_t, z^u_{t+1}])`` also tries to, behind a
    gradient reversal layer, so the encoder strips action information from ``z^u``. SIGReg keeps the whole latent from
    collapsing. Planning compares only ``z^c`` with the goal:

    ``loss = ||[g(z^c, a), h(z^u)] - z_{t+1}||^2 + alpha ||psi - a||^2 + beta ||phi(GRL) - a||^2 + lamb * sigreg``.

    Byrd, Wright and Ha, "CF-JEPA: Improving Robustness of JEPA World Models via Controllability Factorization", 2026.
    Paper: https://arxiv.org/abs/2610.00727
    Code: not released; this follows the paper.

    Args:
        encoder: module mapping images ``(N, C, H, W)`` to embeddings ``(N, embed_dim)`` (ViT-tiny CLS in the paper).
        action_dim: dimension of one action, times the frameskip when actions are stacked.
        ctrl_dim: dimensions of the controllable part ``z^c`` (128 of 192 in the paper); the rest is ``z^u``.
        unctrl_depth: depth of the action-free predictor ``h`` (half of ``depth`` by default).
        alpha: weight of the inverse-dynamics loss on ``z^c`` (3 on Two-Room, 10 on Reacher and Cube in the paper).
        beta: weight of the adversarial action loss on ``z^u``.
        reversal: scale of the reversed gradient that reaches the encoder from the adversary.
    """

    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int,
        embed_dim: int = 192,
        ctrl_dim: int = 128,
        history: int = 3,
        depth: int = 6,
        unctrl_depth: int | None = None,
        num_heads: int = 16,
        head_dim: int = 64,
        mlp_dim: int = 2048,
        proj_dim: int = 2048,
        inv_dim: int = 512,
        dropout: float = 0.1,
        alpha: float = 3.0,
        beta: float = 1.0,
        lamb: float = 0.09,
        reversal: float = 1.0,
        sigreg: SIGReg | None = None,
    ):
        super().__init__()
        if not 0 < ctrl_dim < embed_dim:
            raise ValueError(f"ctrl_dim must be in (0, {embed_dim}), not {ctrl_dim}")
        self.history, self.ctrl_dim = history, ctrl_dim
        self.alpha, self.beta, self.lamb, self.reversal = alpha, beta, lamb, reversal
        c, u = ctrl_dim, embed_dim - ctrl_dim
        self.encoder = encoder
        self.projector = MLP([embed_dim, proj_dim, embed_dim], norm=nn.BatchNorm1d)
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, action_dim), MLP([action_dim, 4 * c, c], act=nn.SiLU))
        block = {"num_heads": num_heads, "mlp_dim": mlp_dim, "head_dim": head_dim, "dropout": dropout}
        self.ctrl_pos = nn.Parameter(torch.randn(1, history, c))
        self.ctrl_predictor = Transformer(c, depth, conditional=True, **block)
        self.ctrl_proj = MLP([c, proj_dim, c], norm=nn.BatchNorm1d)
        self.unctrl_pos = nn.Parameter(torch.randn(1, history, u))
        self.unctrl_predictor = Transformer(u, unctrl_depth or max(1, depth // 2), **block)
        self.unctrl_proj = MLP([u, proj_dim, u], norm=nn.BatchNorm1d)
        self.inverse = MLP([2 * c, inv_dim, action_dim])
        self.adversary = MLP([2 * u, inv_dim, action_dim])
        self.sigreg = sigreg or SIGReg()

    def encode(self, pixels: Tensor) -> Tensor:
        """``(B, T, C, H, W)`` to ``(B, T, embed_dim)``, controllable dims first."""
        return self.projector(self.encoder(pixels.flatten(0, 1))).unflatten(0, pixels.shape[:2])

    def split(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """``z^c`` and ``z^u`` of latents ``(..., embed_dim)``."""
        return z[..., : self.ctrl_dim], z[..., self.ctrl_dim :]

    def predict(self, z: Tensor, actions: Tensor) -> Tensor:
        """Next latents ``(B, T, embed_dim)``: ``z^c`` from ``z^c`` and the actions, ``z^u`` from ``z^u`` alone."""
        (B, T), (zc, zu) = z.shape[:2], self.split(z)
        c = self.ctrl_predictor(zc + self.ctrl_pos[:, :T], self.action_encoder(actions), is_causal=True)
        u = self.unctrl_predictor(zu + self.unctrl_pos[:, :T], is_causal=True)
        return torch.cat([self.ctrl_proj(c.flatten(0, 1)), self.unctrl_proj(u.flatten(0, 1))], -1).unflatten(0, (B, T))

    def cost(self, z: Tensor, goal: Tensor) -> Tensor:
        """Squared distance between the final ``z^c`` of each rollout ``(B, T, D)`` and the goal's ``(B, D)``."""
        return (self.split(z[:, -1])[0] - self.split(goal)[0]).square().mean(1)

    def forward(self, pixels: Tensor, actions: Tensor) -> dict[str, Tensor]:
        """``pixels`` ``(B, history + 1, C, H, W)`` and ``actions`` ``(B, history + 1, action_dim)``."""
        z, H = self.encode(pixels), self.history
        pred = self.predict(z[:, :H], actions[:, :H])
        mse = (pred - z[:, 1 : H + 1]).square().mean()
        (zc, zu), a = self.split(z[:, : H + 1]), actions[:, :H]
        inverse = (self.inverse(torch.cat([zc[:, :-1], zc[:, 1:]], -1)) - a).square().mean()
        pair = GradientReversal.apply(torch.cat([zu[:, :-1], zu[:, 1:]], -1), self.reversal)
        adversary = (self.adversary(pair) - a).square().mean()
        sigreg = self.sigreg(z.transpose(0, 1))
        loss = mse + self.alpha * inverse + self.beta * adversary + self.lamb * sigreg
        return {
            "loss": loss,
            "pred_loss": mse,
            "inverse_loss": inverse,
            "adversary_loss": adversary,
            "sigreg_loss": sigreg,
        }
