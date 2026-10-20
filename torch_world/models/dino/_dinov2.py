"""DINOv2 self-distillation.

Reference: facebookresearch/dinov2 (Apache 2.0; see licenses/). The objective follows dinov2/train/ssl_meta_arch.py,
the losses dinov2/loss and the head dinov2/layers/dino_head.py.
"""

import copy
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm

from torch_world.losses.koleo import koleo
from torch_world.nn.transformer import MLP, Transformer, init_weights
from torch_world.nn.vision_transformer import PatchEmbed
from torch_world.utils.ema import ema_update


class DINOv2Encoder(nn.Module):
    """The ViT of DINOv2: a [cls] token, learned position embeddings resized to each input, and a mask token for iBOT.

    Images ``(B, C, H, W)`` of any size divisible by ``patch_size`` become normalized tokens ``(B, 1 + N, D)``,
    [cls] first. Position embeddings are learned for ``img_size`` and resized bicubically to other grids, such as
    those of local crops. Blocks are the library's pre-norm blocks, without LayerScale.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.patch_size, self.embed_dim = patch_size, embed_dim
        self.grid = (img_size // patch_size, img_size // patch_size)
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.grid[0] * self.grid[1], embed_dim))
        self.transformer = Transformer(embed_dim, depth, num_heads, drop_path, mlp_dim=int(embed_dim * mlp_ratio))
        init_weights(self, init_std)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)

    def positions(self, grid: tuple[int, int]) -> Tensor:
        """Position embeddings ``(1, 1 + h * w, D)`` for a ``grid`` of patches, [cls] first."""
        cls, patches = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        if grid != self.grid:
            patches = patches.unflatten(1, self.grid).permute(0, 3, 1, 2)
            patches = F.interpolate(patches, size=grid, mode="bicubic").flatten(2).transpose(1, 2)
        return torch.cat([cls, patches], dim=1)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """``mask`` ``(B, N)`` is True for patches whose embedding the mask token replaces."""
        grid = (x.size(-2) // self.patch_size, x.size(-1) // self.patch_size)
        x = self.patch_embed(x)
        if mask is not None:
            x = torch.where(mask[..., None], self.mask_token.to(x.dtype), x)
        x = torch.cat([self.cls_token.expand(len(x), -1, -1), x], dim=1)
        return self.transformer(x + self.positions(grid))


class DINOHead(nn.Module):
    """The DINO head: an MLP to a bottleneck, L2 normalization, then weight-normalized prototypes.

    Maps ``(..., in_dim)`` to prototype scores ``(..., num_prototypes)``. DINOv2 scores [cls] tokens (DINO) and patch
    tokens (iBOT) with the same head.
    """

    def __init__(
        self,
        in_dim: int,
        num_prototypes: int = 65536,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        depth: int = 3,
    ):
        super().__init__()
        self.mlp = MLP([in_dim, *[hidden_dim] * (depth - 1), bottleneck_dim])
        init_weights(self.mlp)
        self.prototypes = weight_norm(nn.Linear(bottleneck_dim, num_prototypes, bias=False))
        self.prototypes.parametrizations.weight.original0.data.fill_(1)

    def forward(self, x: Tensor) -> Tensor:
        return self.prototypes(F.normalize(self.mlp(x), dim=-1))


@torch.no_grad()
def sinkhorn_knopp(scores: Tensor, temperature: float, iterations: int = 3) -> Tensor:
    """Teacher targets ``(N, K)`` from prototype scores: each row a distribution, each prototype used equally.

    The sharpened scores ``exp(scores / temperature)`` are rescaled alternately so that every prototype gets mass
    ``1 / K`` and every sample ``1 / N``; rows are then returned summing to one. This is the centering of SwAV and
    DINOv2, in place of DINO's moving-average center.
    """
    q = torch.exp((scores.float() - scores.max()) / temperature)
    q /= q.sum()
    n, k = q.shape
    for _ in range(iterations):
        q /= q.sum(dim=0, keepdim=True) * k
        q /= q.sum(dim=1, keepdim=True) * n
    return q * n


class DINOv2(nn.Module):
    """DINOv2: a student matches an EMA teacher's prototype assignments, image-level (DINO) and patch-level (iBOT).

    The teacher reads the ``G`` global crops of each image; the student reads them with masked patches and the ``V``
    local crops. Three terms:

    - **DINO**: every student [cls] predicts the teacher's centered assignment of every other global crop
      (cross-entropy, averaged over the ``G * (G + V) - G`` pairs).
    - **iBOT**: each masked patch predicts the teacher's assignment of the same, unmasked patch, averaged per image
      and then over all global crops.
    - **KoLeo**: spreads the student's global [cls] tokens within each crop's batch, weighted by ``koleo_weight``.

    Teacher scores are centered by Sinkhorn-Knopp and sharpened by ``teacher_temp``, the student's by
    ``student_temp``. Call ``update_teacher(momentum)`` after each optimizer step (reference: a cosine schedule from
    0.992 to 1); the reference warms ``teacher_temp`` linearly from 0.04 to 0.07.

    Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", TMLR 2024.
    Paper: https://arxiv.org/abs/2304.07193
    Code: https://github.com/facebookresearch/dinov2

    Example::

        encoder = DINOv2Encoder(img_size=224, patch_size=14, embed_dim=384, depth=12, num_heads=6)
        model = DINOv2(encoder, DINOHead(384))
        mask = BlockwiseMask((16, 16))(2 * batch_size).view(2, batch_size, -1)
        loss = model(global_crops, local_crops, mask, teacher_temp=0.07)["loss"]
    """

    def __init__(
        self,
        encoder: DINOv2Encoder,
        head: DINOHead,
        ibot_weight: float = 1.0,
        koleo_weight: float = 0.1,
        student_temp: float = 0.1,
        sinkhorn_iterations: int = 3,
    ):
        super().__init__()
        self.encoder, self.head = encoder, head
        self.teacher_encoder = copy.deepcopy(encoder).requires_grad_(False)
        self.teacher_head = copy.deepcopy(head).requires_grad_(False)
        self.ibot_weight, self.koleo_weight = ibot_weight, koleo_weight
        self.student_temp = student_temp
        self.sinkhorn_iterations = sinkhorn_iterations

    def update_teacher(self, momentum: float):
        ema_update(self.teacher_encoder, self.encoder, momentum)
        ema_update(self.teacher_head, self.head, momentum)

    def encode(self, x: Tensor) -> Tensor:
        """Image embedding ``(B, D)``: the student's [cls] token."""
        return self.encoder(x)[:, 0]

    @torch.no_grad()
    def targets(self, global_crops: Tensor, mask: Tensor, teacher_temp: float) -> tuple[Tensor, Tensor]:
        """Teacher assignments of the global crops' [cls] tokens ``(G, B, K)`` and of the masked patches ``(M, K)``."""
        tokens = self.teacher_encoder(global_crops.flatten(0, 1))
        cls = sinkhorn_knopp(self.teacher_head(tokens[:, 0]), teacher_temp, self.sinkhorn_iterations)
        patches = tokens[:, 1:][mask.flatten(0, 1)]
        if len(patches):
            patches = sinkhorn_knopp(self.teacher_head(patches), teacher_temp, self.sinkhorn_iterations)
        return cls.unflatten(0, global_crops.shape[:2]), patches

    def forward(
        self, global_crops: Tensor, local_crops: Tensor | Sequence[Tensor], mask: Tensor, teacher_temp: float = 0.07
    ) -> dict[str, Tensor]:
        """``global_crops`` ``(G, B, C, H, W)``; ``local_crops`` ``(V, B, C, h, w)`` or ``V`` tensors ``(B, C, h, w)``;
        ``mask`` ``(G, B, N)`` is True for the masked patches of each global crop."""
        G, B = global_crops.shape[:2]
        teacher_cls, teacher_patches = self.targets(global_crops, mask, teacher_temp)

        mask = mask.flatten(0, 1)
        tokens = self.encoder(global_crops.flatten(0, 1), mask)
        local_cls = torch.stack([self.encoder(v)[:, 0] for v in local_crops])
        global_cls = tokens[:, 0].unflatten(0, (G, B))
        student = F.log_softmax(self.head(torch.cat([global_cls, local_cls])) / self.student_temp, dim=-1)

        cross_entropy = -(teacher_cls[:, None] * student[None]).sum(-1).mean(-1)  # (G, G + V) pairs
        other_view = ~torch.eye(G, len(student), dtype=torch.bool, device=student.device)
        dino = cross_entropy[other_view].mean()

        patches = F.log_softmax(self.head(tokens[:, 1:][mask]) / self.student_temp, dim=-1)
        per_image = 1 / mask.sum(1, keepdim=True).clamp(min=1)
        ibot = -((teacher_patches * patches).sum(-1) * per_image.expand_as(mask)[mask]).sum() / len(mask)

        spread = sum(koleo(c) for c in global_cls)
        loss = dino + self.ibot_weight * ibot + self.koleo_weight * spread
        return {"loss": loss, "dino_loss": dino, "ibot_loss": ibot, "koleo_loss": spread}
