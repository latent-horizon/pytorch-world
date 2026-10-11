import torch
from torch import nn


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, momentum: float):
    """``target <- momentum * target + (1 - momentum) * online``, parameter-wise."""
    for t, o in zip(target.parameters(), online.parameters()):
        t.lerp_(o.detach(), 1.0 - momentum)


def linear_schedule(start: float, end: float, step: int, total: int) -> float:
    return start + (end - start) * min(step, total) / total
