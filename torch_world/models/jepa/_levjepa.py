from collections.abc import Sequence

import torch
from torch import Tensor, nn

from torch_world.losses.sigreg import SIGReg
from torch_world.masks.utils import apply_masks
from torch_world.models.base import JEPA
from torch_world.nn.positional import grid_coords
from torch_world.nn.vision_transformer import VisionTransformer


def block_causal_mask(frames: Tensor) -> Tensor:
    """Boolean ``(B, 1, 1 + N, 1 + N)`` attention mask for a [cls] token followed by patches in frames ``(B, N)``.

    Patches attend every patch of their own frame and of earlier frames. [cls] attends every token, and no patch
    attends [cls], so nothing from a later frame reaches an earlier one through it.
    """
    B, N = frames.shape
    mask = frames.new_zeros(B, 1 + N, 1 + N, dtype=torch.bool)
    mask[:, 0] = True
    mask[:, 1:, 1:] = frames[:, :, None] >= frames[:, None, :]
    return mask[:, None]


class LeVJEPAEncoder(VisionTransformer):
    """The video ViT of LeVJEPA: a [cls] readout, 3D RoPE, random token dropping and block-causal attention.

    Videos are ``(B, C, T, H, W)``, with one token per frame by default (``tubelet_size=1``). RoPE positions come from
    the input's own grid, so global and local views of different sizes share the encoder. In training, a fraction
    ``token_drop`` of patch tokens is dropped uniformly at random after patch embedding; in evaluation every token is
    kept. With ``block_causal``, patches attend bidirectionally within a frame and causally across frames, so a clip
    can be encoded incrementally. ``forward`` returns ``(B, 1 + N, D)`` tokens, [cls] first.
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int = 16,
        num_frames: int = 16,
        tubelet_size: int = 1,
        token_drop: float = 0.95,
        block_causal: bool = True,
        init_std: float = 0.02,
        **kwargs,
    ):
        super().__init__(
            img_size,
            patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            cls_token=True,
            use_rope=True,
            init_std=init_std,
            **kwargs,
        )
        nn.init.trunc_normal_(self.cls_token, std=init_std)
        self.token_drop = token_drop
        self.block_causal = block_causal

    def forward(self, x: Tensor, masks: list[Tensor] | None = None) -> Tensor:
        """``masks`` are ``(B, K)`` indices of patches to keep; by default random in training and all in evaluation."""
        grid = tuple(s // k for s, k in zip(x.shape[2:], self.patch_embed.proj.kernel_size))
        x = self.patch_embed(x)
        B, N, _ = x.shape
        pos = grid_coords(grid).to(x.device).expand(B, -1, -1)
        if masks is None and self.training and self.token_drop > 0:
            keep = max(1, round(N * (1 - self.token_drop)))
            masks = [torch.rand(B, N, device=x.device).argsort(dim=1)[:, :keep]]
        if masks is not None:
            x, pos = apply_masks(x, masks), apply_masks(pos, masks)
        attn_mask = block_causal_mask(pos[..., 0]) if self.block_causal and len(grid) == 3 else None
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
        pos = torch.cat([pos.new_zeros(pos.size(0), 1, pos.size(-1)), pos], dim=1)
        return self.transformer(x, attn_mask=attn_mask, pos=pos)


class LeVJEPA(JEPA):
    """LeVJEPA: LeJEPA for video, with one global view, ``V`` local views and no teacher, stop-gradient or predictor.

    Every view of a clip is encoded by the same encoder, read out at its [cls] token and projected. Each projection
    predicts the projection of the global view, and SIGReg pushes the projections of every view toward an isotropic
    Gaussian: ``prediction + lamb * sigreg``. Gradients flow through both sides of the prediction term. Use it with a
    ``LeVJEPAEncoder``, or any encoder whose output starts with a [cls] token.

    Kuhn et al., "LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics", 2026.
    Paper: https://arxiv.org/abs/2608.27395
    Code: https://github.com/galilai-group/levjepa
    """

    def __init__(self, encoder: nn.Module, projector: nn.Module, lamb: float = 0.02, sigreg: SIGReg | None = None):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.lamb = lamb
        self.sigreg = sigreg or SIGReg()

    def encode(self, x: Tensor) -> Tensor:
        """Clip embedding ``(B, D)``: the encoder's [cls] token, before the projector."""
        return self.encoder(x)[:, 0]

    def forward(self, global_view: Tensor, local_views: Tensor | Sequence[Tensor]) -> dict[str, Tensor]:
        """``global_view`` is ``(B, C, T, H, W)``; ``local_views`` is ``(B, V, C, T, h, w)`` or ``V`` tensors ``(B, C, T, h, w)``."""
        if isinstance(local_views, Tensor):
            B, V = local_views.shape[:2]
            local = self.encode(local_views.flatten(0, 1)).unflatten(0, (B, V)).transpose(0, 1)
        else:
            local = torch.stack([self.encode(v) for v in local_views])
        h = torch.cat([self.encode(global_view)[None], local])
        z = self.projector(h.flatten(0, 1)).unflatten(0, h.shape[:2])
        pred = (z[0] - z).square().mean()
        sigreg = self.sigreg(z)
        return {"loss": pred + self.lamb * sigreg, "pred_loss": pred, "sigreg_loss": sigreg}
