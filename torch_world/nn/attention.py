import torch.nn.functional as F
from torch import Tensor, nn

from torch_world.nn.positional import rope


class Attention(nn.Module):
    """Multi-head self-attention on ``F.scaled_dot_product_attention`` with optional rotary positions."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int | None = None,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        use_rope: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim or dim // num_heads
        self.dropout = dropout
        self.use_rope = use_rope
        inner = self.num_heads * self.head_dim
        self.qkv = nn.Linear(dim, 3 * inner, bias=qkv_bias)
        self.proj = nn.Linear(inner, dim)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        pos: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        """``attn_mask`` is boolean (``True`` attends) or additive; ``pos`` holds RoPE coordinates ``(N, A)`` or ``(B, N, A)``."""
        B, N, _ = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        if self.use_rope:
            pos = pos[:, None] if pos.dim() == 3 else pos
            q, k = rope(q, pos), rope(k, pos)
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal
        )
        return self.proj(x.transpose(1, 2).reshape(B, N, -1))
