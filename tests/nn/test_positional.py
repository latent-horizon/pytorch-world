import numpy as np
import torch

from torch_world.nn import grid_coords, rope, sincos_1d, sincos_2d, sincos_3d


def reference_1d(dim, pos):
    omega = 1.0 / 10000 ** (np.arange(dim // 2, dtype=float) / (dim / 2.0))
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def reference_2d(dim, size):
    grid_w, grid_h = np.meshgrid(np.arange(size, dtype=float), np.arange(size, dtype=float))
    return np.concatenate([reference_1d(dim // 2, grid_h), reference_1d(dim // 2, grid_w)], axis=1)


def reference_3d(dim, size, depth):
    grid_h, grid_d, grid_w = np.meshgrid(
        np.arange(size, dtype=float), np.arange(depth, dtype=float), np.arange(size, dtype=float)
    )
    sub = int(np.ceil(dim / 6) * 2)
    out = np.concatenate([reference_1d(sub, grid_d), reference_1d(sub, grid_h), reference_1d(sub, grid_w)], axis=1)
    return out[:, :dim]


def test_sincos_matches_reference():
    assert torch.allclose(sincos_1d(16, torch.arange(7)), torch.tensor(reference_1d(16, np.arange(7.0))).float())
    assert torch.allclose(sincos_2d(32, (5, 5)), torch.tensor(reference_2d(32, 5)).float())
    assert torch.allclose(sincos_3d(40, (3, 4, 4)), torch.tensor(reference_3d(40, 4, 3)).float())


def test_grid_coords_row_major():
    coords = grid_coords((2, 3))
    assert coords.tolist() == [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]


def test_rope_preserves_norm_and_is_relative():
    q, k = torch.randn(2, 1, 1, 12)
    pos = lambda t, h: torch.tensor([[t, h, 0.0]])
    dot = lambda a, b: (rope(q[None], a) * rope(k[None], b)).sum()
    assert torch.allclose(rope(q[None], pos(3, 5)).norm(), q.norm())
    assert torch.allclose(dot(pos(2, 1), pos(5, 4)), dot(pos(10, 7), pos(13, 10)), atol=1e-5)
    assert torch.equal(rope(q[None], torch.zeros(1, 3)), q[None])


def test_rope_leaves_remainder_dims():
    x = torch.randn(1, 4, 10)
    out = rope(x, torch.rand(4, 3) * 5)
    assert torch.equal(out[..., 6:], x[..., 6:])
