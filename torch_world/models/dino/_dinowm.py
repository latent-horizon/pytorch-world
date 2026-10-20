import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_world.masks.utils import frame_causal_mask
from torch_world.models.base import WorldModel
from torch_world.nn.transformer import Transformer


class DINOWM(WorldModel):
    """DINO-WM: a world model over the patch features of a frozen pre-trained encoder (DINOv2).

    Proprioception and action embeddings are tiled over every patch and concatenated on the feature dim. A ViT
    predictor with frame-level causal attention predicts the next frame's visual and proprio features, trained with
    MSE against detached targets. Latents from ``encode`` are ``(B, T, P, D + proprio_emb_dim)``.

    Zhou et al., "DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning", ICML 2025.
    Paper: https://arxiv.org/abs/2411.04983
    Code: https://github.com/gaoyuezhou/dino_wm

    Args:
        encoder: frozen module mapping images ``(N, C, H, W)`` to patch tokens ``(N, num_patches, embed_dim)``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_patches: int,
        embed_dim: int,
        action_dim: int,
        proprio_dim: int = 0,
        action_emb_dim: int = 10,
        proprio_emb_dim: int = 10,
        history: int = 3,
        depth: int = 6,
        num_heads: int = 16,
        head_dim: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.history = history
        self.num_patches = num_patches
        self.action_emb_dim = action_emb_dim
        self.encoder = encoder.requires_grad_(False).eval()
        self.action_encoder = nn.Linear(action_dim, action_emb_dim)
        self.proprio_encoder = nn.Linear(proprio_dim, proprio_emb_dim) if proprio_dim else None
        dim = embed_dim + action_emb_dim + (proprio_emb_dim if proprio_dim else 0)
        self.pos_embed = nn.Parameter(torch.randn(1, history * num_patches, dim))
        self.predictor = Transformer(
            dim, depth, num_heads, mlp_dim=mlp_dim, head_dim=head_dim, qkv_bias=False, dropout=dropout
        )
        self.register_buffer("attn_mask", frame_causal_mask(history, num_patches), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def encode(self, pixels: Tensor, proprio: Tensor | None = None) -> Tensor:
        """``pixels`` ``(B, T, C, H, W)`` and ``proprio`` ``(B, T, proprio_dim)`` to ``(B, T, P, D')``."""
        with torch.no_grad():
            z = self.encoder(pixels.flatten(0, 1)).unflatten(0, pixels.shape[:2])
        if self.proprio_encoder is not None:
            z = torch.cat([z, self._tile(self.proprio_encoder(proprio))], dim=-1)
        return z

    def _tile(self, x: Tensor) -> Tensor:
        return x[:, :, None].expand(-1, -1, self.num_patches, -1)

    def predict(self, z: Tensor, actions: Tensor) -> Tensor:
        T, P = z.shape[1:3]
        x = torch.cat([z, self._tile(self.action_encoder(actions))], dim=-1).flatten(1, 2)
        x = self.predictor(x + self.pos_embed[:, : T * P], attn_mask=self.attn_mask[: T * P, : T * P])
        return x.unflatten(1, (T, P))[..., : -self.action_emb_dim]

    def forward(self, pixels: Tensor, actions: Tensor, proprio: Tensor | None = None) -> dict[str, Tensor]:
        """``pixels`` ``(B, history + 1, C, H, W)``, ``actions`` ``(B, history + 1, action_dim)``."""
        z = self.encode(pixels, proprio)
        pred = self.predict(z[:, : self.history], actions[:, : self.history])
        target = z[:, 1 : self.history + 1].detach()
        out = {"loss": F.mse_loss(pred, target)}
        if self.proprio_encoder is not None:
            d = self.proprio_encoder.out_features
            out["visual_loss"] = F.mse_loss(pred[..., :-d], target[..., :-d])
            out["proprio_loss"] = F.mse_loss(pred[..., -d:], target[..., -d:])
        return out
