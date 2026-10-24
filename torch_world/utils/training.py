import itertools
from collections.abc import Callable, Iterable

import torch
from torch import nn

from torch_world.utils.batchnorm import recompute_batchnorm


def fit(
    model: nn.Module,
    loader: Iterable,
    epochs: int,
    lr: float = 5e-4,
    weight_decay: float = 1e-3,
    inputs: Callable | None = None,
    clip: float | None = None,
    step: Callable[[], None] | None = None,
) -> dict[str, list[float]]:
    """Train ``model`` with AdamW for ``epochs`` over ``loader``; returns every logged loss, one value per step.

    ``inputs(batch)`` gives the model's positional arguments, ``(batch["pixels"], batch["actions"])`` by default. Every
    output named ``loss`` or ``*_loss`` is logged. ``clip`` clips the gradient norm, and ``step()`` runs after every
    optimizer step, for example an EMA target update. At the end, BatchNorm statistics are recomputed on 50 batches
    and the model is put in eval mode, ready to plan with.
    """
    inputs = inputs or (lambda batch: (batch["pixels"], batch["actions"]))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    log: dict[str, list[float]] = {}
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
                    log.setdefault(k.removesuffix("_loss"), []).append(v.item())
    if any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        recompute_batchnorm(model, itertools.islice(loader, 50), lambda batch: model(*inputs(batch)))
    model.eval()
    return log
