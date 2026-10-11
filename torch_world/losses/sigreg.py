import torch
from torch import Tensor, nn


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization.

    Projects embeddings on ``num_slices`` random unit directions and averages the Epps-Pulley statistic,
    which measures each 1D projection's distance from a standard normal by comparing characteristic
    functions on ``knots`` trapezoid points in ``[0, t_max]``.

    Balestriero & LeCun, "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics", 2025.
    Paper: https://arxiv.org/abs/2511.08544
    Code: https://github.com/galilai-group/lejepa
    """

    def __init__(self, num_slices: int = 1024, knots: int = 17, t_max: float = 3.0):
        super().__init__()
        self.num_slices = num_slices
        t = torch.linspace(0, t_max, knots)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        phi = (-t.square() / 2).exp()
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, x: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """``x`` is ``(..., N, D)`` with samples on dim ``-2``; returns the statistic averaged over slices and leading dims."""
        A = torch.randn(x.size(-1), self.num_slices, device=x.device, dtype=x.dtype, generator=generator)
        A = A / A.norm(dim=0)
        xt = (x @ A).unsqueeze(-1) * self.t
        err = (xt.cos().mean(-3) - self.phi).square() + xt.sin().mean(-3).square()
        return ((err @ self.weights) * x.size(-2)).mean()
