from collections.abc import Callable, Iterable

import torch
from torch import nn


@torch.no_grad()
def recompute_batchnorm(model: nn.Module, inputs: Iterable, forward: Callable):
    """Re-estimate BatchNorm running statistics as exact averages of ``forward(x)`` over ``inputs``.

    Running averages lag behind features that move during training; call this before evaluating in eval mode.
    """
    norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    momenta = [m.momentum for m in norms]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None
    training = model.training
    model.train()
    for x in inputs:
        forward(x)
    for m, momentum in zip(norms, momenta):
        m.momentum = momentum
    model.train(training)
