import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_world.losses.rdmreg import RDMReg
from torch_world.models.base import WorldModel
from torch_world.nn.transformer import MLP, Transformer


def rep_relu(x: Tensor) -> Tensor:
    """ReLU on the forward pass with the GELU gradient on the backward pass, so zeroed units keep learning."""
    gelu = F.gelu(x)
    return gelu - gelu.detach() + x.relu().detach()


class RepReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return rep_relu(x)


class LpWM(WorldModel):
    """LpWM: a LeWM-style JEPA world model whose latents are non-negative and sparse, predicted under actions.

    Each frame is encoded to one embedding (the encoder's CLS token), projected and passed through a RepReLU link, so
    codes are exactly zero on most units. A causal transformer, conditioned on action embeddings through adaLN-zero,
    predicts the next code from the last ``history`` frames, through the same link. RDMReg pulls the codes of each
    timestep toward a rectified Laplace, which prevents collapse and sets the sparsity:
    ``loss = mse + lamb * rdmreg``. Which units are active encodes the discrete regime (for example the room), their
    magnitudes the continuous state within it. Planning minimizes the distance to the goal code (CEM).

    Kuang et al., "LpWM: A Case for Sparse Representations in World Models", 2026.
    Paper: https://arxiv.org/abs/2608.22764
    Code: https://github.com/YilunKuang/lpworldmodel

    Args:
        encoder: module mapping images ``(N, C, H, W)`` to embeddings ``(N, embed_dim)`` (ViT-tiny CLS in the paper).
        action_dim: dimension of one action, times the frameskip when actions are stacked.
    """

    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int,
        embed_dim: int = 192,
        history: int = 3,
        depth: int = 6,
        num_heads: int = 16,
        head_dim: int = 64,
        mlp_dim: int = 2048,
        proj_dim: int = 2048,
        dropout: float = 0.1,
        lamb: float = 10.0,
        rdmreg: RDMReg | None = None,
    ):
        super().__init__()
        self.history = history
        self.lamb = lamb
        self.encoder = encoder
        self.projector = nn.Sequential(MLP([embed_dim, proj_dim, embed_dim], norm=nn.BatchNorm1d), RepReLU())
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, action_dim), MLP([action_dim, 4 * embed_dim, embed_dim], act=nn.SiLU)
        )
        self.pos_embed = nn.Parameter(torch.randn(1, history, embed_dim))
        self.predictor = Transformer(
            embed_dim, depth, num_heads, conditional=True, mlp_dim=mlp_dim, head_dim=head_dim, dropout=dropout
        )
        self.pred_proj = nn.Sequential(MLP([embed_dim, proj_dim, embed_dim], norm=nn.BatchNorm1d), RepReLU())
        self.rdmreg = rdmreg or RDMReg()

    def encode(self, pixels: Tensor) -> Tensor:
        """``(B, T, C, H, W)`` to non-negative sparse codes ``(B, T, embed_dim)``."""
        return self.projector(self.encoder(pixels.flatten(0, 1))).unflatten(0, pixels.shape[:2])

    def predict(self, z: Tensor, actions: Tensor) -> Tensor:
        T = z.size(1)
        h = self.predictor(z + self.pos_embed[:, :T], self.action_encoder(actions), is_causal=True)
        return self.pred_proj(h.flatten(0, 1)).unflatten(0, (z.size(0), T))

    def forward(self, pixels: Tensor, actions: Tensor) -> dict[str, Tensor]:
        """``pixels`` ``(B, history + 1, C, H, W)`` and ``actions`` ``(B, history + 1, action_dim)``."""
        z = self.encode(pixels)
        pred = self.predict(z[:, : self.history], actions[:, : self.history])
        mse = (pred - z[:, 1 : self.history + 1]).square().mean()
        rdmreg = self.rdmreg(z)
        return {"loss": mse + self.lamb * rdmreg, "pred_loss": mse, "rdmreg_loss": rdmreg}
