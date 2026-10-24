import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RDMReg(nn.Module):
    """Rectified Distribution-Matching Regularization: sliced 2-Wasserstein distance to a rectified product prior.

    Embeddings and samples of the target ``ReLU(mu + sigma * GN_p)`` (each coordinate i.i.d., ``sigma`` giving
    ``GN_p`` unit variance) are projected on ``num_slices`` random unit directions; each 1D projection is sorted over
    samples and matched by squared error. ``p = 1`` (a rectified Laplace) yields non-negative codes with exact zeros;
    ``p = 2`` without ``rectify`` recovers the isotropic Gaussian of LeJEPA/LeWM.

    Kuang et al., "LpWM: A Case for Sparse Representations in World Models", 2026.
    Paper: https://arxiv.org/abs/2608.22764
    Code: https://github.com/YilunKuang/lpworldmodel
    """

    def __init__(self, num_slices: int = 1024, p: float = 1.0, mu: float = 0.0, rectify: bool = True):
        super().__init__()
        self.num_slices, self.p, self.mu, self.rectify = num_slices, p, mu, rectify
        self.sigma = math.sqrt(math.gamma(1 / p) / math.gamma(3 / p)) / p ** (1 / p)

    def sample(self, shape: tuple[int, ...], device=None, dtype=None) -> Tensor:
        """Target samples: ``|x| = (p g)^(1/p)`` with ``g ~ Gamma(1/p, 1)`` and a random sign is ``GN_p``."""
        g = torch.distributions.Gamma(torch.tensor(1 / self.p, device=device), 1.0).sample(shape)
        sign = torch.randint(2, shape, device=device) * 2 - 1
        x = self.mu + self.sigma * sign * (self.p * g).pow(1 / self.p)
        return (x.relu() if self.rectify else x).to(dtype)

    def forward(self, z: Tensor) -> Tensor:
        """``z`` is ``(N, ..., D)`` with samples on dim ``0``; middle dims are matched separately and averaged."""
        directions = F.normalize(torch.randn(z.size(-1), self.num_slices, device=z.device, dtype=z.dtype), dim=0)
        target = self.sample(z.shape, z.device, z.dtype)
        return ((z @ directions).sort(0).values - (target @ directions).sort(0).values).square().mean()
