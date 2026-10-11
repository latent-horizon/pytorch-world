import torch
from torch import Tensor, nn

from torch_world.masks.utils import apply_masks
from torch_world.nn.positional import grid_coords, sincos
from torch_world.nn.transformer import Transformer, init_weights, rescale_blocks

VIT_SIZES = {
    "tiny": (192, 12, 3),
    "small": (384, 12, 6),
    "base": (768, 12, 12),
    "large": (1024, 24, 16),
    "huge": (1280, 32, 16),
    "giant": (1408, 40, 16),
}


def _pair(x):
    return tuple(x) if isinstance(x, (tuple, list)) else (x, x)


class PatchEmbed(nn.Module):
    """Images ``(B, C, H, W)`` or videos ``(B, C, T, H, W)`` to patch tokens ``(B, N, D)``."""

    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768, tubelet_size: int | None = None):
        super().__init__()
        if tubelet_size is None:
            self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)
        else:
            kernel = (tubelet_size, patch_size, patch_size)
            self.proj = nn.Conv3d(in_chans, embed_dim, kernel, kernel)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    """ViT for images or videos (``num_frames > 1`` uses ``tubelet_size`` frames per token).

    Positions are fixed sin-cos embeddings, or 2D/3D RoPE when ``use_rope``. ``forward`` takes optional
    index masks and then encodes only the kept tokens, as JEPA context encoders do.
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
        num_frames: int = 1,
        tubelet_size: int = 2,
        cls_token: bool = False,
        use_rope: bool = False,
        init_std: float = 0.02,
    ):
        super().__init__()
        h, w = (s // patch_size for s in _pair(img_size))
        video = num_frames > 1
        self.grid = (num_frames // tubelet_size, h, w) if video else (h, w)
        self.embed_dim = embed_dim
        self.use_rope = use_rope
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim, tubelet_size if video else None)
        pos_embed = torch.zeros(len(grid_coords(self.grid)), embed_dim) if use_rope else sincos(embed_dim, self.grid)
        self.register_buffer("pos_embed", pos_embed[None], persistent=False)
        self.register_buffer("coords", grid_coords(self.grid), persistent=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if cls_token else None
        self.transformer = Transformer(
            embed_dim,
            depth,
            num_heads,
            drop_path,
            mlp_dim=int(embed_dim * mlp_ratio),
            qkv_bias=qkv_bias,
            use_rope=use_rope,
        )
        init_weights(self, init_std)
        rescale_blocks(self.transformer.blocks)

    @classmethod
    def from_size(cls, size: str, **kwargs) -> "VisionTransformer":
        """Build a standard ViT: ``tiny``, ``small``, ``base``, ``large``, ``huge`` or ``giant``."""
        embed_dim, depth, num_heads = VIT_SIZES[size]
        return cls(embed_dim=embed_dim, depth=depth, num_heads=num_heads, **kwargs)

    @property
    def num_patches(self) -> int:
        return self.pos_embed.size(1)

    def forward(self, x: Tensor, masks: list[Tensor] | None = None) -> Tensor:
        x = self.patch_embed(x) + self.pos_embed
        B, pos = x.size(0), None
        if masks is not None:
            x = apply_masks(x, masks)
        if self.use_rope:
            pos = self.coords if masks is None else apply_masks(self.coords.expand(B, -1, -1), masks)
        if self.cls_token is not None:
            x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
            if pos is not None:
                pos = torch.cat([pos.new_zeros(pos.shape[:-2] + (1, pos.size(-1))), pos], dim=-2)
        return self.transformer(x, pos=pos)
