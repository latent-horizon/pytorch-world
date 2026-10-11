import math

import torch
from torch import Tensor


def _uniform(bounds: tuple[float, float], u: float) -> float:
    return bounds[0] + u * (bounds[1] - bounds[0])


class MultiBlockMask:
    """I-JEPA multi-block masking on a 2D patch grid.

    Samples ``num_pred`` target blocks and ``num_enc`` context blocks per image. Context blocks avoid the
    targets unless ``allow_overlap``. Block sizes are shared across the batch, and every mask is cropped
    to the shortest one so masks stack into ``(B, K)`` index tensors.
    """

    def __init__(
        self,
        grid: tuple[int, int],
        enc_scale: tuple[float, float] = (0.85, 1.0),
        pred_scale: tuple[float, float] = (0.15, 0.2),
        aspect_ratio: tuple[float, float] = (0.75, 1.5),
        num_enc: int = 1,
        num_pred: int = 4,
        min_keep: int = 10,
        allow_overlap: bool = False,
    ):
        self.height, self.width = grid
        self.enc_scale, self.pred_scale, self.aspect_ratio = enc_scale, pred_scale, aspect_ratio
        self.num_enc, self.num_pred = num_enc, num_pred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap

    def _block_size(self, scale, aspect_ratio, g) -> tuple[int, int]:
        u = torch.rand(1, generator=g).item()
        keep = int(self.height * self.width * _uniform(scale, u))
        ar = _uniform(aspect_ratio, u)
        h = min(round(math.sqrt(keep * ar)), self.height - 1)
        w = min(round(math.sqrt(keep / ar)), self.width - 1)
        return h, w

    def _block(self, size, g, regions: list[Tensor] | None = None) -> tuple[Tensor, Tensor]:
        h, w = size
        if h * w <= self.min_keep:
            raise ValueError(f"a {h}x{w} block never keeps more than min_keep={self.min_keep} patches")
        tries, timeout = 0, 20
        while True:
            top = torch.randint(0, self.height - h, (1,), generator=g).item()
            left = torch.randint(0, self.width - w, (1,), generator=g).item()
            block = torch.zeros(self.height, self.width, dtype=torch.bool)
            block[top : top + h, left : left + w] = True
            mask = block.clone()
            for region in (regions or [])[: max(len(regions or []) - tries, 0)]:
                mask &= region
            idx = mask.flatten().nonzero().squeeze(-1)
            if len(idx) > self.min_keep:
                return idx, ~block
            timeout -= 1
            if timeout == 0:
                tries, timeout = tries + 1, 20

    def __call__(self, batch_size: int, generator: torch.Generator | None = None) -> tuple[list[Tensor], list[Tensor]]:
        """Return ``(masks_enc, masks_pred)``: lists of ``num_enc`` and ``num_pred`` index tensors of shape ``(B, K)``."""
        g = generator
        p_size = self._block_size(self.pred_scale, self.aspect_ratio, g)
        e_size = self._block_size(self.enc_scale, (1.0, 1.0), g)
        enc, pred = [], []
        for _ in range(batch_size):
            p, complements = zip(*(self._block(p_size, g) for _ in range(self.num_pred)))
            regions = None if self.allow_overlap else list(complements)
            e = [self._block(e_size, g, regions)[0] for _ in range(self.num_enc)]
            pred.append(p)
            enc.append(e)
        return _stack(enc, self.num_enc), _stack(pred, self.num_pred)


class MultiBlock3DMask:
    """V-JEPA spatio-temporal multi-block masking on a ``(T, H, W)`` token grid.

    The union of ``num_blocks`` tubes is the prediction target; the remaining tokens are the context.
    """

    def __init__(
        self,
        grid: tuple[int, int, int],
        spatial_scale: tuple[float, float] = (0.15, 0.15),
        temporal_scale: tuple[float, float] = (1.0, 1.0),
        aspect_ratio: tuple[float, float] = (0.75, 1.5),
        num_blocks: int = 8,
        max_temporal_keep: float = 1.0,
    ):
        self.duration, self.height, self.width = grid
        self.spatial_scale, self.temporal_scale, self.aspect_ratio = spatial_scale, temporal_scale, aspect_ratio
        self.num_blocks = num_blocks
        self.max_context_duration = max(1, int(self.duration * max_temporal_keep))

    def _block_size(self, g) -> tuple[int, int, int]:
        u = torch.rand(3, generator=g).tolist()
        t = max(1, int(self.duration * _uniform(self.temporal_scale, u[0])))
        keep = int(self.height * self.width * _uniform(self.spatial_scale, u[1]))
        ar = _uniform(self.aspect_ratio, u[2])
        h = min(round(math.sqrt(keep * ar)), self.height)
        w = min(round(math.sqrt(keep / ar)), self.width)
        return t, h, w

    def _context(self, size, g) -> Tensor:
        t, h, w = size
        context = torch.ones(self.duration, self.height, self.width, dtype=torch.bool)
        for _ in range(self.num_blocks):
            top = torch.randint(0, self.height - h + 1, (1,), generator=g).item()
            left = torch.randint(0, self.width - w + 1, (1,), generator=g).item()
            start = torch.randint(0, self.duration - t + 1, (1,), generator=g).item()
            context[start : start + t, top : top + h, left : left + w] = False
        context[self.max_context_duration :] = False
        return context.flatten()

    def __call__(self, batch_size: int, generator: torch.Generator | None = None) -> tuple[list[Tensor], list[Tensor]]:
        """Return ``([mask_enc], [mask_pred])`` with index tensors of shape ``(B, K)``."""
        size = self._block_size(generator)
        enc, pred = [], []
        while len(enc) < batch_size:
            context = self._context(size, generator)
            if context.any():
                enc.append(context.nonzero().squeeze(-1))
                pred.append((~context).nonzero().squeeze(-1))
        return [_crop(enc)], [_crop(pred)]


def _crop(masks: list[Tensor]) -> Tensor:
    k = min(len(m) for m in masks)
    return torch.stack([m[:k] for m in masks])


def _stack(per_sample: list, n: int) -> list[Tensor]:
    return [_crop([s[i] for s in per_sample]) for i in range(n)]
