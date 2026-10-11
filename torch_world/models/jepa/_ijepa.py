import torch
import torch.nn.functional as F
from torch import Tensor

from torch_world.masks.utils import apply_masks, repeat_interleave_batch
from torch_world.models.base import JEPA
from torch_world.nn.predictor import MaskedPredictor
from torch_world.nn.vision_transformer import VisionTransformer


class IJEPA(JEPA):
    """I-JEPA: predict the target encoder's features of masked image blocks from a visible context block.

    Targets are layer-normalized features of an EMA target encoder, and the loss is smooth L1. Call
    ``update_target(momentum)`` after each optimizer step (reference schedule: 0.996 to 1.0).

    Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture", CVPR 2023.
    Paper: https://arxiv.org/abs/2301.08243
    Code: https://github.com/facebookresearch/ijepa

    Example::

        encoder = VisionTransformer.from_size("huge", img_size=224, patch_size=14)
        model = IJEPA(encoder, MaskedPredictor(encoder.grid, encoder.embed_dim, 384, depth=12, num_heads=12))
        masks_enc, masks_pred = MultiBlockMask(encoder.grid)(batch_size)
        loss = model(images, masks_enc, masks_pred)["loss"]
    """

    def __init__(self, encoder: VisionTransformer, predictor: MaskedPredictor):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.target_encoder = self.make_target_encoder(encoder)

    @torch.no_grad()
    def targets(self, x: Tensor, masks_enc: list[Tensor], masks_pred: list[Tensor]) -> Tensor:
        h = self.target_encoder(x)
        h = F.layer_norm(h, h.shape[-1:])
        return repeat_interleave_batch(apply_masks(h, masks_pred), len(x), len(masks_enc))

    def forward(self, x: Tensor, masks_enc: list[Tensor], masks_pred: list[Tensor]) -> dict[str, Tensor]:
        h = self.targets(x, masks_enc, masks_pred)
        z = self.predictor(self.encoder(x, masks_enc), masks_enc, masks_pred)
        return {"loss": F.smooth_l1_loss(z, h)}
