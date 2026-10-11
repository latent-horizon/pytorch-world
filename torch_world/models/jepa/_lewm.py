import torch
from torch import Tensor, nn

from torch_world.losses.sigreg import SIGReg
from torch_world.models.base import WorldModel
from torch_world.nn.transformer import MLP, Transformer


class LeWM(WorldModel):
    """LeWorldModel: an end-to-end JEPA world model trained from pixels with two terms, next-embedding MSE and SIGReg.

    Each frame is encoded to one embedding (the encoder's CLS token) and projected. A causal transformer, conditioned
    on action embeddings through adaLN-zero, predicts the next embedding from the last ``history`` frames. There is no
    EMA target and no stop-gradient: ``loss = mse + lamb * sigreg``.

    Maes et al., "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels", 2026.
    Paper: https://arxiv.org/abs/2603.19312
    Code: https://github.com/lucas-maes/le-wm

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
        lamb: float = 0.09,
        sigreg: SIGReg | None = None,
    ):
        super().__init__()
        self.history = history
        self.lamb = lamb
        self.encoder = encoder
        self.projector = MLP([embed_dim, proj_dim, embed_dim], norm=nn.BatchNorm1d)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, action_dim), MLP([action_dim, 4 * embed_dim, embed_dim], act=nn.SiLU)
        )
        self.pos_embed = nn.Parameter(torch.randn(1, history, embed_dim))
        self.predictor = Transformer(
            embed_dim, depth, num_heads, conditional=True, mlp_dim=mlp_dim, head_dim=head_dim, dropout=dropout
        )
        self.pred_proj = MLP([embed_dim, proj_dim, embed_dim], norm=nn.BatchNorm1d)
        self.sigreg = sigreg or SIGReg()

    def encode(self, pixels: Tensor) -> Tensor:
        """``(B, T, C, H, W)`` to ``(B, T, embed_dim)``."""
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
        sigreg = self.sigreg(z.transpose(0, 1))
        return {"loss": mse + self.lamb * sigreg, "pred_loss": mse, "sigreg_loss": sigreg}
