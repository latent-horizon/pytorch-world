import torch
from torch import Tensor


def _smooth_rank(values: Tensor, eps: float = 1e-7) -> float:
    p = values / values.sum() + eps
    return (-(p * p.log()).sum()).exp().item()


@torch.no_grad()
def rankme(z: Tensor, center: bool = False, eps: float = 1e-7) -> float:
    """Effective rank of embeddings ``(N, D)``: the exponential of the entropy of their normalized singular values.

    Ranges from 1 (collapsed) to ``min(N, D)``, needs no labels, and tracks downstream accuracy across
    hyperparameters of one method. As in the paper, the embeddings are used as they are; ``center`` subtracts their
    mean first, for features dominated by one component shared by every input (pooled tokens of mostly-empty
    images), which would otherwise read as rank 1.

    Garrido et al., "RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by
    Their Rank", ICML 2023.
    Paper: https://arxiv.org/abs/2210.02885
    """
    z = z.double()
    return _smooth_rank(torch.linalg.svdvals(z - z.mean(0) if center else z), eps)


@torch.no_grad()
def lidar(z: Tensor, delta: float = 1e-4, eps: float = 1e-7) -> float:
    """Effective rank of the linear discriminant analysis matrix of embeddings ``(N, Q, D)``: ``N`` surrogate classes
    (for example images) with ``Q >= 2`` samples each (for example augmented views).

    The between-class covariance is whitened by the within-class covariance, so only directions that separate
    classes count. Designed for JEPAs, where it predicts linear-probe accuracy better than RankMe.

    Thilak et al., "LiDAR: Sensing Linear Probing Performance in Joint Embedding SSL Architectures", ICLR 2024.
    Paper: https://arxiv.org/abs/2312.04000
    """
    z = z.double()
    n, q, d = z.shape
    means = z.mean(1)
    between = torch.cov(means.T, correction=1).view(d, d)
    centered = (z - means[:, None]).reshape(-1, d)
    within = centered.T @ centered / (n * (q - 1)) + delta * torch.eye(d, dtype=z.dtype, device=z.device)
    values, vectors = torch.linalg.eigh(within)
    inv_sqrt = vectors @ torch.diag(values.rsqrt()) @ vectors.T
    return _smooth_rank(torch.linalg.eigvalsh(inv_sqrt @ between @ inv_sqrt).clamp_min(0), eps)
