import torch
import torch.nn.functional as F
from torch import Tensor


def info_nce(a: Tensor, b: Tensor, temperature: float | Tensor = 0.07, symmetric: bool = True) -> Tensor:
    """InfoNCE between paired rows of ``a`` and ``b``, both ``(B, D)``; averaged over both directions when ``symmetric``."""
    logits = F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T / temperature
    labels = torch.arange(len(a), device=a.device)
    loss = F.cross_entropy(logits, labels)
    return (loss + F.cross_entropy(logits.T, labels)) / 2 if symmetric else loss
