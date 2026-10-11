from collections.abc import Callable
from functools import partial
from itertools import pairwise

import torch
from torch import Tensor, nn

from torch_world.nn.attention import Attention

LayerNorm = partial(nn.LayerNorm, eps=1e-6)


class MLP(nn.Sequential):
    """Linear layers through ``dims`` with ``norm``, ``act`` and dropout between them."""

    def __init__(
        self,
        dims: list[int],
        norm: Callable[[int], nn.Module] | None = None,
        act: Callable[[], nn.Module] = nn.GELU,
        dropout: float = 0.0,
    ):
        layers = []
        for i, (a, b) in enumerate(pairwise(dims)):
            layers.append(nn.Linear(a, b))
            if i < len(dims) - 2:
                layers += [norm(b)] if norm else []
                layers += [act(), nn.Dropout(dropout)] if dropout else [act()]
        super().__init__(*layers)


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = x.new_empty((x.size(0),) + (1,) * (x.dim() - 1)).bernoulli_(1 - self.p)
        return x * keep / (1 - self.p)


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int | None = None,
        head_dim: int | None = None,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        use_rope: bool = False,
        act: Callable[[], nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, head_dim, qkv_bias, dropout, use_rope)
        self.norm2 = LayerNorm(dim)
        self.mlp = MLP([dim, mlp_dim or 4 * dim, dim], act=act, dropout=dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None, pos: Tensor | None = None, is_causal: bool = False):
        x = x + self.drop_path(self.attn(self.norm1(x), attn_mask, pos, is_causal))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class AdaLNBlock(Block):
    """Transformer block conditioned by adaptive layer norm with zero-initialized gates.

    Peebles & Xie, "Scalable Diffusion Models with Transformers", ICCV 2023.
    Paper: https://arxiv.org/abs/2212.09748
    """

    def __init__(self, dim: int, num_heads: int, **kwargs):
        super().__init__(dim, num_heads, **kwargs)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self, x: Tensor, c: Tensor, attn_mask: Tensor | None = None, pos: Tensor | None = None, is_causal: bool = False
    ):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(c).chunk(6, dim=-1)
        x = x + gate1 * self.drop_path(self.attn(self.norm1(x) * (1 + scale1) + shift1, attn_mask, pos, is_causal))
        return x + gate2 * self.drop_path(self.mlp(self.norm2(x) * (1 + scale2) + shift2))


class Transformer(nn.Module):
    """Stack of ``Block`` (or ``AdaLNBlock`` when ``conditional``) followed by a final layer norm."""

    def __init__(
        self, dim: int, depth: int, num_heads: int, drop_path: float = 0.0, conditional: bool = False, **kwargs
    ):
        super().__init__()
        block = AdaLNBlock if conditional else Block
        rates = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(block(dim, num_heads, drop_path=r, **kwargs) for r in rates)
        self.norm = LayerNorm(dim)

    def forward(self, x: Tensor, c: Tensor | None = None, **kwargs) -> Tensor:
        for block in self.blocks:
            x = block(x, **kwargs) if c is None else block(x, c, **kwargs)
        return self.norm(x)


def init_weights(module: nn.Module, std: float = 0.02):
    """Truncated-normal linear weights, zero biases and unit norms, as in the JEPA reference code."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def rescale_blocks(blocks: nn.ModuleList):
    """Divide residual output projections of layer ``i`` by ``sqrt(2 i)``."""
    with torch.no_grad():
        for i, block in enumerate(blocks, start=1):
            block.attn.proj.weight.div_((2.0 * i) ** 0.5)
            block.mlp[-1].weight.div_((2.0 * i) ** 0.5)
