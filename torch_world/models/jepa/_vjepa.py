import torch
import torch.nn.functional as F
from torch import Tensor

from torch_world.masks.utils import apply_masks
from torch_world.models.base import JEPA
from torch_world.nn.predictor import MaskedPredictor
from torch_world.nn.vision_transformer import VisionTransformer


class VJEPA(JEPA):
    """V-JEPA: predict EMA target features of masked spatio-temporal tubes of a video.

    ``masks_enc`` and ``masks_pred`` hold one ``(B, K)`` context/target pair per masking strategy (short-range and
    long-range in the paper), and strategy ``i`` uses the predictor's mask token ``i``. The loss is
    ``mean(|z - h| ** loss_exp) / loss_exp``, averaged over strategies, plus ``reg_coeff`` times a variance hinge on
    the predictions. V-JEPA 2 keeps this objective with 3D RoPE: build the encoder and predictor with ``use_rope=True``.

    Bardes et al., "Revisiting Feature Prediction for Learning Visual Representations from Video", 2024.
    Paper: https://arxiv.org/abs/2404.08471
    Code: https://github.com/facebookresearch/jepa

    Assran et al., "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning", 2025.
    Paper: https://arxiv.org/abs/2506.09985
    Code: https://github.com/facebookresearch/vjepa2
    """

    def __init__(
        self, encoder: VisionTransformer, predictor: MaskedPredictor, loss_exp: float = 1.0, reg_coeff: float = 0.0
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.target_encoder = self.make_target_encoder(encoder)
        self.loss_exp = loss_exp
        self.reg_coeff = reg_coeff

    def forward(self, x: Tensor, masks_enc: list[Tensor], masks_pred: list[Tensor]) -> dict[str, Tensor]:
        with torch.no_grad():
            h = self.target_encoder(x)
            h = F.layer_norm(h, h.shape[-1:])
        jepa, reg = 0.0, 0.0
        for i, (m_enc, m_pred) in enumerate(zip(masks_enc, masks_pred)):
            z = self.predictor(self.encoder(x, [m_enc]), [m_enc], [m_pred], mask_index=i)
            jepa = jepa + (z - apply_masks(h, [m_pred])).abs().pow(self.loss_exp).mean() / self.loss_exp
            reg = reg + F.relu(1.0 - (z.var(dim=1) + 1e-4).sqrt()).mean()
        jepa, reg = jepa / len(masks_pred), reg / len(masks_pred)
        return {"loss": jepa + self.reg_coeff * reg, "jepa_loss": jepa, "reg_loss": reg}
