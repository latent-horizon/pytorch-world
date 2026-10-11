import torch
from torch import Tensor


def apply_masks(x: Tensor, masks: list[Tensor]) -> Tensor:
    """Gather the tokens indexed by each ``(B, K)`` mask from ``x`` of shape ``(B, N, ...)``; stack masks on the batch dim."""
    out = []
    for m in masks:
        index = m.view(m.shape + (1,) * (x.dim() - 2)).expand(*m.shape, *x.shape[2:])
        out.append(torch.gather(x, 1, index))
    return torch.cat(out, dim=0)


def repeat_interleave_batch(x: Tensor, batch_size: int, repeat: int) -> Tensor:
    """Repeat each consecutive chunk of ``batch_size`` rows ``repeat`` times."""
    chunks = x.split(batch_size)
    return torch.cat([c for c in chunks for _ in range(repeat)], dim=0)


def frame_causal_mask(num_frames: int, tokens_per_frame: int, window: int | None = None, device=None) -> Tensor:
    """Boolean ``(N, N)`` mask where a token attends every token of its own frame and of the ``window`` frames before it."""
    t = torch.arange(num_frames, device=device).repeat_interleave(tokens_per_frame)
    lag = t[:, None] - t[None, :]
    return (lag >= 0) & (lag < (window or num_frames))
