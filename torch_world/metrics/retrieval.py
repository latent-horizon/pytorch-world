import torch
from torch import Tensor


def topk_accuracy(logits: Tensor, labels: Tensor, k: int = 1) -> float:
    """Fraction of rows of ``logits`` ``(N, C)`` whose ``k`` highest scores include the label, as in zero-shot
    classification."""
    return (logits.topk(k, dim=1).indices == labels[:, None]).any(1).float().mean().item()


def recall_at_k(similarity: Tensor, k: int | tuple[int, ...] = (1, 5, 10), targets: Tensor | None = None):
    """Retrieval recall@k for a query-by-candidate ``similarity`` ``(Q, C)``: the fraction of queries whose correct
    candidate (``targets`` ``(Q,)``, by default the diagonal) ranks among the top ``k``. Returns a float, or a dict for
    several ``k``, as VL-JEPA reports for text-to-video and video-to-text retrieval (transpose for the other way).
    """
    targets = torch.arange(len(similarity), device=similarity.device) if targets is None else targets
    correct = similarity.gather(1, targets[:, None])
    rank = (similarity > correct).sum(1)
    if isinstance(k, int):
        return (rank < k).float().mean().item()
    return {f"R@{n}": (rank < n).float().mean().item() for n in k}
