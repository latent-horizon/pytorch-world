import math
from itertools import pairwise

import torch
from torch import Tensor


def _uniform(low: float, high: float, g: torch.Generator | None) -> float:
    return low + (high - low) * torch.rand(1, generator=g).item()


def _randint(high: int, g: torch.Generator | None) -> int:
    """A uniform integer in ``[0, high]``."""
    return int(torch.randint(high + 1, (1,), generator=g))


class BlockwiseMask:
    """Block-wise masking for iBOT, as DINOv2 samples it: boolean ``(B, N)`` masks over a 2D patch grid.

    ``int(probability * B)`` of the ``B`` images are masked and the rest are not. Their mask ratios are spread over
    ``ratio``: the ``i``-th masked image draws its ratio uniformly from the ``i``-th of equal bins. A mask grows by
    rectangles of at least ``min_patches`` patches, at most ``max_fraction`` of the grid, and an aspect ratio
    log-uniform in ``[min_aspect, 1 / min_aspect]``, as in BEiT. Masked and unmasked images come in random order.

    Reference: dinov2/data/masking.py and dinov2/data/collate.py in facebookresearch/dinov2.
    """

    def __init__(
        self,
        grid: tuple[int, int],
        ratio: tuple[float, float] = (0.1, 0.5),
        probability: float = 0.5,
        min_patches: int = 4,
        max_fraction: float = 0.5,
        min_aspect: float = 0.3,
    ):
        self.height, self.width = grid
        self.ratio, self.probability = ratio, probability
        self.min_patches = min_patches
        self.max_patches = int(max_fraction * self.height * self.width)
        self.log_aspect = (math.log(min_aspect), -math.log(min_aspect))

    def _add_block(self, mask: Tensor, limit: int, g: torch.Generator | None) -> int:
        """Mask one rectangle that adds between 1 and ``limit`` patches, in up to 10 tries; return how many it added."""
        for _ in range(10):
            area = _uniform(self.min_patches, limit, g)
            aspect = math.exp(_uniform(*self.log_aspect, g))
            h, w = round(math.sqrt(area * aspect)), round(math.sqrt(area / aspect))
            if h >= self.height or w >= self.width:
                continue
            top, left = _randint(self.height - h, g), _randint(self.width - w, g)
            block = mask[top : top + h, left : left + w]
            added = h * w - int(block.sum())
            if 0 < added <= limit:
                block.fill_(True)
                return added
        return 0

    def _mask(self, count: int, g: torch.Generator | None) -> Tensor:
        mask = torch.zeros(self.height, self.width, dtype=torch.bool)
        masked = 0
        while masked < count:
            added = self._add_block(mask, min(count - masked, self.max_patches), g)
            if added == 0:
                break
            masked += added
        return mask

    def __call__(self, batch_size: int, generator: torch.Generator | None = None) -> Tensor:
        num_masked = int(batch_size * self.probability)
        bounds = torch.linspace(*self.ratio, num_masked + 1).tolist()
        num_patches = self.height * self.width
        masks = [self._mask(int(num_patches * _uniform(lo, hi, generator)), generator) for lo, hi in pairwise(bounds)]
        masks += [torch.zeros(self.height, self.width, dtype=torch.bool)] * (batch_size - num_masked)
        order = torch.randperm(batch_size, generator=generator)
        return torch.stack(masks)[order].flatten(1)
