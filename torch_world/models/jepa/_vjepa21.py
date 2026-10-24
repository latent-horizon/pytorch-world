import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_world.masks.utils import apply_masks, repeat_interleave_batch
from torch_world.models.base import JEPA
from torch_world.nn.predictor import MaskedPredictor
from torch_world.nn.transformer import MLP, LayerNorm, init_weights
from torch_world.nn.vision_transformer import PatchEmbed, VisionTransformer


class VJEPA21Encoder(VisionTransformer):
    """The video ViT of V-JEPA 2.1: multi-level outputs and, with ``images``, a separate image tokenizer.

    ``levels`` evenly spaced blocks, the last one the final block (blocks ``[2, 5, 8, 11]`` of a 12-block ViT), each
    have their own layer norm. ``forward(x, all_levels=True)`` concatenates their outputs into ``(B, N, levels * D)``,
    the input of the predictor and the target of the loss; by default it returns only the normalized final output, as
    a V-JEPA encoder does. With ``images``, images ``(B, C, H, W)`` go through a 2D patch embedding onto the first
    frame of the grid instead of being repeated over time, and a learned modality embedding marks images and videos.
    """

    def __init__(self, *args, levels: int = 4, images: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        depth = len(self.transformer.blocks)
        self.levels = [(i + 1) * depth // levels - 1 for i in range(levels)]
        self.norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in self.levels[:-1]] + [self.transformer.norm])
        conv = self.patch_embed.proj
        self.image_embed = PatchEmbed(conv.kernel_size[-1], conv.in_channels, self.embed_dim) if images else None
        self.modality_embed = nn.Parameter(1e-6 * torch.randn(2, 1, self.embed_dim)) if images else None

    def forward(self, x: Tensor, masks: list[Tensor] | None = None, all_levels: bool = False) -> Tensor:
        image = self.image_embed is not None and x.dim() == 4
        if image:
            x = self.image_embed(x)
            x, coords = x + self.pos_embed[:, : x.size(1)], self.coords[: x.size(1)]
        else:
            x, coords = self.embed(x)
        if self.modality_embed is not None:
            x = x + self.modality_embed[int(image)]
        pos = None
        if masks is not None:
            x = apply_masks(x, masks)
        if self.use_rope:
            pos = coords if masks is None else apply_masks(coords.expand(x.size(0), -1, -1), masks)
        outs = []
        for i, block in enumerate(self.transformer.blocks):
            x = block(x, pos=pos)
            if i in self.levels:
                outs.append(self.norms[self.levels.index(i)](x))
        return torch.cat(outs, dim=-1) if all_levels else outs[-1]


class VJEPA21Predictor(MaskedPredictor):
    """V-JEPA 2.1 predictor: reads and predicts ``levels`` encoder levels, for target and context tokens alike.

    An MLP fuses the concatenated levels of the context tokens. ``forward`` returns ``(z_pred, z_context)``: the
    predictions at the target positions and, through a separate head, at the context positions, each with
    ``levels * embed_dim`` channels. With ``images``, a learned modality embedding is added to every token.
    """

    def __init__(
        self,
        grid: tuple[int, ...],
        embed_dim: int,
        predictor_dim: int = 384,
        levels: int = 4,
        images: bool = False,
        init_std: float = 0.02,
        **kwargs,
    ):
        super().__init__(grid, embed_dim, predictor_dim, init_std=init_std, **kwargs)
        self.embed = MLP([levels * embed_dim, embed_dim, predictor_dim])
        self.proj = nn.Linear(predictor_dim, levels * embed_dim)
        self.proj_context = nn.Linear(predictor_dim, levels * embed_dim)
        for m in (self.embed, self.proj, self.proj_context):
            init_weights(m, init_std)
        self.modality_embed = nn.Parameter(1e-6 * torch.randn(2, 1, 1, predictor_dim)) if images else None

    def forward(
        self, x: Tensor, masks_x: list[Tensor], masks_y: list[Tensor], mask_index: int = 0, image: bool = False
    ) -> tuple[Tensor, Tensor]:
        B = x.size(0) // len(masks_x)
        pos_embed = self.pos_embed.expand(B, -1, -1)
        x = self.embed(x) + apply_masks(pos_embed, masks_x)
        y = self.mask_tokens[mask_index] + repeat_interleave_batch(apply_masks(pos_embed, masks_y), B, len(masks_x))
        n = x.size(1)
        x = torch.cat([x.repeat(len(masks_y), 1, 1), y], dim=1)
        if self.modality_embed is not None:
            x = x + self.modality_embed[int(image)]
        pos = None
        if self.use_rope:
            coords = self.coords.expand(B, -1, -1)
            pos_x = apply_masks(coords, masks_x).repeat(len(masks_y), 1, 1)
            pos_y = repeat_interleave_batch(apply_masks(coords, masks_y), B, len(masks_x))
            pos = torch.cat([pos_x, pos_y], dim=1)
        x = self.transformer(x, pos=pos)
        return self.proj(x[:, n:]), self.proj_context(x[:, :n])


class VJEPA21(JEPA):
    """V-JEPA 2.1: V-JEPA 2 with a dense loss on every token and deep self-supervision of several encoder levels.

    The predictor reads the context tokens' encoder levels and predicts the EMA target's levels, each layer-normalized,
    at both the masked tokens and the visible context tokens. With ``p = loss_exp``, the loss is

    ``mean(|z - h| ** p) / p  +  context_coeff * mean(w * |z_ctx - h_ctx| ** p) / p``

    averaged over masking strategies, where ``w = 1 / sqrt(d)`` and ``d`` is the distance on the token grid from a
    context token to the nearest masked token (``w = 1`` without ``distance_weighting``). The paper warms
    ``context_coeff`` up linearly from 0 and uses 0.5 for videos and 0.7 for images; pass ``context_coeff`` to
    ``forward`` to schedule it. Images ``(B, C, H, W)`` train the same way when the encoder has ``images=True``.

    Mur-Labadia et al., "V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning", 2026.
    Paper: https://arxiv.org/abs/2603.14482
    Code: https://github.com/facebookresearch/vjepa2
    """

    def __init__(
        self,
        encoder: VJEPA21Encoder,
        predictor: VJEPA21Predictor,
        context_coeff: float = 0.5,
        distance_weighting: bool = True,
        loss_exp: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.target_encoder = self.make_target_encoder(encoder)
        self.context_coeff = context_coeff
        self.distance_weighting = distance_weighting
        self.loss_exp = loss_exp

    def distance_weights(self, mask_enc: Tensor, mask_pred: Tensor) -> Tensor:
        """``1 / sqrt(d)`` ``(B, K)``, ``d`` the grid distance from each context token to its nearest target token."""
        coords = self.encoder.coords.expand(mask_enc.size(0), -1, -1)
        d = torch.cdist(apply_masks(coords, [mask_enc]), apply_masks(coords, [mask_pred])).amin(-1)
        return d.clamp_min(1.0).rsqrt()

    def forward(
        self, x: Tensor, masks_enc: list[Tensor], masks_pred: list[Tensor], context_coeff: float | None = None
    ) -> dict[str, Tensor]:
        L, D, p = len(self.encoder.levels), self.encoder.embed_dim, self.loss_exp
        image = x.dim() == 4
        with torch.no_grad():
            h = self.target_encoder(x, all_levels=True).unflatten(-1, (L, D))
            h = F.layer_norm(h, (D,)).flatten(-2)
        pred, context = 0.0, 0.0
        for i, (m_enc, m_pred) in enumerate(zip(masks_enc, masks_pred)):
            z, c = self.predictor(self.encoder(x, [m_enc], all_levels=True), [m_enc], [m_pred], i, image=image)
            pred = pred + (z - apply_masks(h, [m_pred])).abs().pow(p).mean() / p
            err = (c - apply_masks(h, [m_enc])).abs().pow(p)
            if self.distance_weighting:
                err = err * self.distance_weights(m_enc, m_pred)[..., None]
            context = context + err.mean() / p
        pred, context = pred / len(masks_pred), context / len(masks_pred)
        coeff = self.context_coeff if context_coeff is None else context_coeff
        return {"loss": pred + coeff * context, "pred_loss": pred, "context_loss": context}
