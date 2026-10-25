import math
from collections.abc import Sequence

import torch
from torch import Tensor


def crps(samples: Tensor, target: Tensor, fair: bool = False) -> Tensor:
    """Continuous ranked probability score of an ensemble ``(B, S, ...)`` against ``target`` ``(B, ...)``, one value
    per element of ``target`` (lower is better).

    ``CRPS = E|X - y| - E|X - X'| / 2`` over the ``S`` samples. A point forecast (``S = 1``) scores its absolute error,
    so deterministic and stochastic models share one scale. ``fair`` uses the unbiased ``E|X - X'|``, which divides by
    ``S (S - 1)`` instead of ``S^2`` and does not reward small ensembles.

    Gneiting and Raftery, "Strictly Proper Scoring Rules, Prediction, and Estimation", JASA 2007.
    Paper: https://doi.org/10.1198/016214506000001437
    """
    x, S = samples.movedim(1, -1).sort(-1).values, samples.size(1)
    skill = (x - target[..., None]).abs().mean(-1)
    weights = 2 * torch.arange(1, S + 1, device=x.device, dtype=x.dtype) - S - 1
    spread = 2 * (x * weights).sum(-1) / (S * (S - 1) if fair and S > 1 else S * S)
    return skill - 0.5 * spread


def energy_score(samples: Tensor, target: Tensor, event_dims: int = 1, fair: bool = False) -> Tensor:
    """Energy score, the multivariate CRPS, of an ensemble ``(B, S, ...)`` against ``target`` ``(B, ...)``; the last
    ``event_dims`` dimensions form one vector, so ``(B, S, T, D)`` gives ``(B, T)`` (lower is better).

    ``ES = E||X - y|| - E||X - X'|| / 2``; a point forecast scores its Euclidean error. ``fair`` as in ``crps``.

    Gneiting and Raftery, "Strictly Proper Scoring Rules, Prediction, and Estimation", JASA 2007.
    Paper: https://doi.org/10.1198/016214506000001437
    """
    S = samples.size(1)
    x = samples.flatten(-event_dims).movedim(1, -2)  # (B, ..., S, E)
    y = target.flatten(-event_dims)[..., None, :]
    skill = (x - y).norm(dim=-1).mean(-1)
    spread = torch.cdist(x.flatten(0, -3), x.flatten(0, -3)).unflatten(0, x.shape[:-2]).sum((-1, -2))
    return skill - 0.5 * spread / (S * (S - 1) if fair and S > 1 else S * S)


def gaussian_nll(mean: Tensor, var: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """Negative log-likelihood of ``target`` under ``N(mean, var)``, one value per element, in nats (lower is better).

    For an ensemble, pass its mean and variance over the samples: the score of the Gaussian that matches its first two
    moments.
    """
    var = var.clamp_min(eps)
    return 0.5 * (math.log(2 * math.pi) + var.log() + (target - mean).square() / var)


def interval_coverage(samples: Tensor, target: Tensor, levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95)) -> Tensor:
    """Whether ``target`` ``(B, ...)`` falls in the central interval of the ensemble ``(B, S, ...)`` at each nominal
    level: ``(len(levels), B, ...)`` of 0 and 1. A calibrated ensemble covers each level's share of the targets; less
    means it is overconfident, more that it is underconfident.

    The interval of level ``l`` runs between the ``(1 - l) / 2`` and ``(1 + l) / 2`` quantiles of the samples,
    interpolated linearly. A point forecast covers nothing.
    """
    x, S = samples.movedim(1, -1).sort(-1).values, samples.size(1)

    def quantile(q: float) -> Tensor:
        pos = q * (S - 1)
        lo, frac = int(pos), pos - int(pos)
        return x[..., lo] + frac * (x[..., min(lo + 1, S - 1)] - x[..., lo])

    inside = [(quantile((1 - l) / 2) <= target) & (target <= quantile((1 + l) / 2)) for l in levels]
    return torch.stack(inside).to(samples.dtype)


def spread_skill(samples: Tensor, target: Tensor, dim: int | tuple[int, ...] | None = None) -> tuple[Tensor, Tensor]:
    """Spread and skill of an ensemble ``(B, S, ...)`` against ``target`` ``(B, ...)``, averaged over ``dim`` of the
    target (all of it by default): ``spread`` is the root mean ensemble variance, scaled by ``(S + 1) / S`` for the
    finite ensemble, and ``skill`` the root mean squared error of the ensemble mean. A calibrated ensemble has a ratio
    ``spread / skill`` near 1 at every horizon; below 1 it is overconfident. A point forecast has zero spread.

    Fortin et al., "Why Should Ensemble Spread Match the RMSE of the Ensemble Mean?", J. Hydrometeorology 2014.
    Paper: https://doi.org/10.1175/JHM-D-14-0008.1
    """
    S = samples.size(1)
    var = samples.var(1, correction=1) * (S + 1) / S if S > 1 else torch.zeros_like(target)
    error = (samples.mean(1) - target).square()
    if dim is None:
        return var.mean().sqrt(), error.mean().sqrt()
    return var.mean(dim).sqrt(), error.mean(dim).sqrt()
