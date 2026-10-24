import torch
from torch import Tensor, nn

from torch_world.losses.sigreg import SIGReg
from torch_world.models.base import WorldModel
from torch_world.models.jepa._hwm import MacroLevel
from torch_world.models.jepa._lewm import LeWM
from torch_world.nn.transformer import MLP


class MacroPool(nn.Module):
    """Pool ``stride`` action embeddings ``(N, stride, dim)`` into a macro-action ``(N, macro_dim)``: a one-layer
    transformer over the chunk, read out at a CLS token, as H-JEPA's sequence encoder."""

    def __init__(self, dim: int, macro_dim: int, stride: int, width: int = 64, num_heads: int = 2):
        super().__init__()
        self.inp = nn.Linear(dim, width)
        self.pos = nn.Parameter(torch.randn(1, stride + 1, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, num_heads, 2 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width), nn.Linear(width, macro_dim))
        self.norm = nn.BatchNorm1d(macro_dim, affine=False)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.cat([self.pos[:, :1].expand(len(x), -1, -1), self.inp(x) + self.pos[:, 1:]], dim=1)
        return self.norm(self.head(self.encoder(x)[:, 0]))


class HJEPA(WorldModel):
    """H-JEPA: a two-level hierarchy of JEPA world models trained end-to-end from pixels.

    Level 1 is a ``LeWM``: next-embedding prediction from primitive actions. Level 2 has its own state encoder, an
    MLP over every ``stride``-th level-1 latent, ``z2_k = E2(z1_{k * stride})``, and its own macro-actions: a
    one-layer transformer pools the ``stride`` level-1 action embeddings between two of its states. Its predictor
    (``MacroLevel``) predicts the next level-2 latent. Each level is a JEPA with teacher-forced MSE and SIGReg on its
    own latents, and the level-2 loss flows into the level-1 encoder and action encoder::

        loss = mse_1 + lamb * sigreg(z1) + mse_2 + lamb * sigreg(z2)

    Planning is top-down (``planning.HierarchicalPlanner``): level 2 plans macro-actions toward ``E2(goal)``, and
    level 1 plans the next ``stride`` actions so that ``E2`` of its last predicted latent matches the first level-2
    prediction, the subgoal, in level-2 space. The paper plans every level by gradient descent
    (``planning.GradientPlanner``).

    Windows hold ``high.history + 1`` level-2 states, ``T = high.history * stride + 1`` frames; level 1 is trained on
    a random crop of ``low.history + 1`` frames, as the reference crops each level to its context.

    Zhang et al., "H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning", 2026.
    Paper: https://arxiv.org/abs/2610.06805
    Code: https://github.com/kevinghst/H-JEPA

    Args:
        low: level 1; its ``lamb`` weights SIGReg at both levels, uniform across levels as in the paper.
        stride: level-1 steps per level-2 step (2 in the paper, 4 here by default).
        macro_dim: dimension of a level-2 macro-action (4 in the paper).
        embed_dim: dimension of level-1 latents and of level-2 latents.
    """

    def __init__(
        self,
        low: LeWM,
        stride: int = 4,
        macro_dim: int = 4,
        embed_dim: int = 192,
        encoder_dim: int = 384,
        history: int = 3,
        depth: int = 4,
        num_heads: int = 8,
        head_dim: int = 24,
        mlp_dim: int = 1024,
        dropout: float = 0.1,
        sigreg: SIGReg | None = None,
    ):
        super().__init__()
        self.low, self.stride = low, stride
        self.history = low.history
        self.state_encoder = MLP([embed_dim, encoder_dim, embed_dim], act=nn.ReLU)
        self.macro_encoder = MacroPool(embed_dim, macro_dim, stride)
        self.high = MacroLevel(embed_dim, macro_dim, history, depth, num_heads, head_dim, mlp_dim, dropout, p=2)
        self.sigreg = sigreg or SIGReg()

    def encode(self, pixels: Tensor) -> Tensor:
        """``(B, T, C, H, W)`` to level-1 latents ``(B, T, embed_dim)``."""
        return self.low.encode(pixels)

    def predict(self, z: Tensor, actions: Tensor) -> Tensor:
        return self.low.predict(z, actions)

    def lift(self, z: Tensor) -> Tensor:
        """Level-1 latents ``(B, T, D)`` to level-2 latents ``(B, T, D)``, step by step (H-JEPA's window of 1)."""
        return self.state_encoder(z)

    def encode_macro(self, actions: Tensor) -> Tensor:
        """Primitive actions ``(B, K * stride, A)`` to macro-actions ``(B, K, macro_dim)``."""
        chunks = self.low.action_encoder(actions).unflatten(1, (-1, self.stride))
        return self.macro_encoder(chunks.flatten(0, 1)).unflatten(0, chunks.shape[:2])

    def forward(self, pixels: Tensor, actions: Tensor) -> dict[str, Tensor]:
        """``pixels`` ``(B, T, C, H, W)`` and ``actions`` ``(B, T, A)`` with ``T = high.history * stride + 1``."""
        z = self.encode(pixels)
        h = self.low.history
        t = int(torch.randint(z.size(1) - h, ()))
        pred = self.low.predict(z[:, t : t + h], actions[:, t : t + h])
        mse_1 = (pred - z[:, t + 1 : t + h + 1]).square().mean()
        sigreg_1 = self.low.sigreg(z[:, t : t + h + 1].transpose(0, 1))
        z2 = self.lift(z[:, :: self.stride])
        macro = self.encode_macro(actions[:, : (z2.size(1) - 1) * self.stride])
        mse_2 = (self.high.predict(z2[:, :-1], macro) - z2[:, 1:]).square().mean()
        sigreg_2 = self.sigreg(z2.transpose(0, 1))
        loss = mse_1 + mse_2 + self.low.lamb * (sigreg_1 + sigreg_2)
        return {
            "loss": loss,
            "low_loss": mse_1,
            "high_loss": mse_2,
            "low_sigreg_loss": sigreg_1,
            "high_sigreg_loss": sigreg_2,
        }
