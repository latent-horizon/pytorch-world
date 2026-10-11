import torch

from torch_world.nn import CLSEncoder, PixelPatches


def test_pixel_patches_round_trip():
    patches = PixelPatches(4)
    x = torch.rand(2, 5, 3, 8, 8)
    tokens = patches(x.flatten(0, 1)).unflatten(0, (2, 5))
    assert tokens.shape == (2, 5, 4, 48) and patches.dim == 48
    assert torch.equal(patches.unpatchify(tokens, (2, 2)), x)


def test_cls_encoder_resizes_crops():
    encoder = CLSEncoder(16, 4, embed_dim=32, depth=1, num_heads=2, resize=True)
    assert encoder(torch.rand(3, 3, 8, 8)).shape == (3, 32)
