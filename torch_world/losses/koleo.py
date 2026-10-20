import torch
import torch.nn.functional as F
from torch import Tensor


def koleo(x: Tensor, eps: float = 1e-8) -> Tensor:
    """KoLeo regularizer on ``(N, D)`` embeddings: minus the mean log distance from each to its nearest neighbor.

    It is the Kozachenko-Leonenko differential-entropy estimator up to constants, computed on L2-normalized rows, so
    minimizing it spreads the embeddings uniformly within the batch.

    Sablayrolles et al., "Spreading vectors for similarity search", ICLR 2019.
    Paper: https://arxiv.org/abs/1806.03198
    """
    x = F.normalize(x, dim=-1, eps=eps)
    with torch.no_grad():
        nearest = (x @ x.T).fill_diagonal_(-1).argmax(dim=1)
    return -torch.log(F.pairwise_distance(x, x[nearest], eps=eps) + eps).mean()
