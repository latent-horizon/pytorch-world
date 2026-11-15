import itertools
from collections.abc import Callable, Iterable

import torch
from torch import nn

from torch_world.utils.batchnorm import recompute_batchnorm
from torch_world.utils.monitor import Monitor


def fit(
    model: nn.Module,
    loader: Iterable,
    epochs: int,
    lr: float = 5e-4,
    weight_decay: float = 1e-3,
    inputs: Callable | None = None,
    clip: float | None = None,
    step: Callable[[], None] | None = None,
    monitor: Monitor | None = None,
) -> dict[str, list[float]]:
    """Train ``model`` with AdamW for ``epochs`` over ``loader``; returns every logged loss, one value per step.

    ``inputs(batch)`` gives the model's positional arguments, ``(batch["pixels"], batch["actions"])`` by default. Every
    output named ``loss`` or ``*_loss`` is logged, and ``monitor``, for example a ``TensorBoardMonitor`` or
    ``RerunMonitor``, receives each of them as a scalar, one call per optimizer step. ``clip`` clips the gradient
    norm, and ``step()`` runs after every optimizer step, for example an EMA target update. At the end, BatchNorm
    statistics are recomputed on 50 batches and the model is put in eval mode, ready to plan with.
    """
    inputs = inputs or (lambda batch: (batch["pixels"], batch["actions"]))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    log: dict[str, list[float]] = {}
    steps = 0
    model.train()
    for _ in range(epochs):
        for batch in loader:
            out = model(*inputs(batch))
            opt.zero_grad()
            out["loss"].backward()
            if clip is not None:
                nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            if step is not None:
                step()
            for k, v in out.items():
                if k == "loss" or k.endswith("_loss"):
                    value = v.item()
                    log.setdefault(k.removesuffix("_loss"), []).append(value)
                    if monitor is not None:
                        monitor.scalar(steps, k.removesuffix("_loss"), value)
            steps += 1
    if any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        recompute_batchnorm(model, itertools.islice(loader, 50), lambda batch: model(*inputs(batch)))
    model.eval()
    return log
