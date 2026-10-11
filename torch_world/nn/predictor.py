import torch
from torch import Tensor, nn

from torch_world.masks.utils import apply_masks, repeat_interleave_batch
from torch_world.nn.positional import grid_coords, sincos
from torch_world.nn.transformer import Transformer, init_weights, rescale_blocks


class MaskedPredictor(nn.Module):
    """JEPA predictor: from context tokens, predict target tokens at masked positions of a patch grid.

    Each target position is a learnable mask token plus its position. ``num_mask_tokens`` holds one token
    per masking strategy, as in V-JEPA.
    """

    def __init__(
        self,
        grid: tuple[int, ...],
        embed_dim: int,
        predictor_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
        num_mask_tokens: int = 1,
        zero_init_mask_tokens: bool = False,
        use_rope: bool = False,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.use_rope = use_rope
        self.embed = nn.Linear(embed_dim, predictor_dim)
        self.mask_tokens = nn.Parameter(torch.zeros(num_mask_tokens, 1, 1, predictor_dim))
        pos_embed = torch.zeros(len(grid_coords(grid)), predictor_dim) if use_rope else sincos(predictor_dim, grid)
        self.register_buffer("pos_embed", pos_embed[None], persistent=False)
        self.register_buffer("coords", grid_coords(grid), persistent=False)
        self.transformer = Transformer(
            predictor_dim,
            depth,
            num_heads,
            drop_path,
            mlp_dim=int(predictor_dim * mlp_ratio),
            qkv_bias=qkv_bias,
            use_rope=use_rope,
        )
        self.proj = nn.Linear(predictor_dim, embed_dim)
        init_weights(self, init_std)
        rescale_blocks(self.transformer.blocks)
        if not zero_init_mask_tokens:
            nn.init.trunc_normal_(self.mask_tokens, std=init_std)

    def forward(self, x: Tensor, masks_x: list[Tensor], masks_y: list[Tensor], mask_index: int = 0) -> Tensor:
        """``x`` is ``(len(masks_x) * B, Nx, D)``; returns ``(len(masks_y) * len(masks_x) * B, Ny, D)``."""
        B = x.size(0) // len(masks_x)
        pos_embed = self.pos_embed.expand(B, -1, -1)
        x = self.embed(x) + apply_masks(pos_embed, masks_x)
        y = self.mask_tokens[mask_index] + repeat_interleave_batch(apply_masks(pos_embed, masks_y), B, len(masks_x))
        n = x.size(1)
        x = torch.cat([x.repeat(len(masks_y), 1, 1), y], dim=1)
        pos = None
        if self.use_rope:
            coords = self.coords.expand(B, -1, -1)
            pos_x = apply_masks(coords, masks_x).repeat(len(masks_y), 1, 1)
            pos_y = repeat_interleave_batch(apply_masks(coords, masks_y), B, len(masks_x))
            pos = torch.cat([pos_x, pos_y], dim=1)
        return self.proj(self.transformer(x, pos=pos)[:, n:])
