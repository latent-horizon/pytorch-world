from collections.abc import Sequence

import torch
from torch import Tensor, nn

from torch_world.losses.sigreg import SIGReg
from torch_world.models.base import JEPA


class LeJEPA(JEPA):
    """LeJEPA: multi-view prediction with SIGReg, with no teacher, stop-gradient or schedules.

    Each view's projection predicts the mean projection of the ``num_global`` global views, and SIGReg pushes the
    projections of every view toward an isotropic Gaussian. ``lamb`` trades the two terms:
    ``(1 - lamb) * prediction + lamb * sigreg``.

    Balestriero & LeCun, "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics", 2025.
    Paper: https://arxiv.org/abs/2511.08544
    Code: https://github.com/galilai-group/lejepa
    """

    def __init__(self, encoder: nn.Module, projector: nn.Module, lamb: float = 0.05, sigreg: SIGReg | None = None):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.lamb = lamb
        self.sigreg = sigreg or SIGReg()

    def forward(self, views: Tensor | Sequence[Tensor], num_global: int | None = None) -> dict[str, Tensor]:
        """``views`` is ``(B, V, ...)`` or a list of ``(B, ...)`` tensors whose first ``num_global`` are global views."""
        views = views.unbind(1) if isinstance(views, Tensor) else views
        z = torch.stack([self.projector(self.encoder(v)) for v in views])
        pred = (z[:num_global].mean(0) - z).square().mean()
        sigreg = self.sigreg(z)
        return {"loss": (1 - self.lamb) * pred + self.lamb * sigreg, "pred_loss": pred, "sigreg_loss": sigreg}
