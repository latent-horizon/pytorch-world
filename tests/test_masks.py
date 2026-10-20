import pytest
import torch

from torch_world.masks import (
    BlockwiseMask,
    MultiBlock3DMask,
    MultiBlockMask,
    apply_masks,
    frame_causal_mask,
    repeat_interleave_batch,
)


def test_apply_masks_and_repeat():
    x = torch.arange(2 * 5).view(2, 5, 1).float()
    out = apply_masks(x, [torch.tensor([[0, 1], [2, 3]]), torch.tensor([[4, 4], [0, 0]])])
    assert out.squeeze(-1).tolist() == [[0, 1], [7, 8], [4, 4], [5, 5]]
    assert repeat_interleave_batch(torch.arange(4), 2, 2).tolist() == [0, 1, 0, 1, 2, 3, 2, 3]


def test_multiblock_context_disjoint_from_targets():
    g = torch.Generator().manual_seed(0)
    gen = MultiBlockMask((14, 14), num_pred=4, min_keep=4)
    enc, pred = gen(8, generator=g)
    assert len(enc) == 1 and len(pred) == 4
    assert all(m.shape[0] == 8 for m in enc + pred)
    for b in range(8):
        targets = set(torch.cat([p[b] for p in pred]).tolist())
        assert not targets & set(enc[0][b].tolist())
    assert int(enc[0].max()) < 196


def test_multiblock_target_size():
    g = torch.Generator().manual_seed(1)
    _, pred = MultiBlockMask((14, 14), pred_scale=(0.15, 0.15), aspect_ratio=(1.0, 1.0))(2, generator=g)
    assert pred[0].shape[1] == 25


def test_multiblock_3d_partitions_tokens():
    g = torch.Generator().manual_seed(0)
    gen = MultiBlock3DMask((4, 8, 8), spatial_scale=(0.15, 0.15), num_blocks=8)
    [enc], [pred] = gen(4, generator=g)
    assert enc.shape[0] == pred.shape[0] == 4
    for b in range(4):
        assert not set(enc[b].tolist()) & set(pred[b].tolist())
    whole = gen.max_context_duration * 64
    assert enc.shape[1] + pred.shape[1] <= 4 * 64 and enc.shape[1] <= whole


def test_frame_causal_mask():
    m = frame_causal_mask(3, 2)
    assert m.shape == (6, 6)
    assert m[0, :2].all() and not m[0, 2:].any() and m[5].all()
    w = frame_causal_mask(3, 1, window=2)
    assert w.int().tolist() == [[1, 0, 0], [1, 1, 0], [0, 1, 1]]


def test_multiblock_rejects_blocks_smaller_than_min_keep():
    with pytest.raises(ValueError, match="min_keep"):
        MultiBlockMask((4, 4), pred_scale=(0.15, 0.15), aspect_ratio=(1.0, 1.0), min_keep=2)(1)


def test_blockwise_mask_masks_a_fraction_of_images_within_the_ratio_range():
    masks = BlockwiseMask((14, 14), ratio=(0.1, 0.5), probability=0.5)(16, generator=torch.Generator().manual_seed(0))
    counts = masks.sum(1)
    assert masks.shape == (16, 196) and masks.dtype == torch.bool
    assert (counts > 0).sum() == 8
    assert counts.max() <= 0.5 * 196


def test_blockwise_mask_is_reproducible():
    mask = BlockwiseMask((8, 8))
    a, b = (mask(6, generator=torch.Generator().manual_seed(3)) for _ in range(2))
    assert torch.equal(a, b)
