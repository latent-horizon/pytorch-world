import importlib.util

import pytest
import torch

from torch_world.nn import ConvVideoAE, VideoVAE


def test_conv_video_ae_shapes_and_training():
    torch.manual_seed(0)
    ae = ConvVideoAE(latent_channels=4, width=16, depth=2)
    pixels = torch.rand(2, 3, 3, 16, 16)
    latents = ae.encode(pixels)
    assert latents.shape == (2, 3, 4, 4, 4) and ae.spatial_ratio == 4 and ae.temporal_ratio == 1
    assert ae.decode(latents).shape == pixels.shape
    optimizer = torch.optim.Adam(ae.parameters(), lr=3e-3)
    first = ae(pixels)["loss"].item()
    for _ in range(30):
        optimizer.zero_grad()
        ae(pixels)["loss"].backward()
        optimizer.step()
    assert ae(pixels)["loss"].item() < first


def test_wan_vae_config_parsing():
    if importlib.util.find_spec("diffusers") is None:
        pytest.skip("needs torch-world[wam]")
    try:
        from diffusers import AutoencoderKLWan
    except RuntimeError as error:  # diffusers imports torchvision, which fails when built for another torch
        pytest.skip(f"diffusers is installed but does not import: {error}")

    vae = VideoVAE(
        AutoencoderKLWan(base_dim=16, z_dim=4, num_res_blocks=1, latents_mean=[0.0] * 4, latents_std=[1.0] * 4),
        [0.0] * 4,
        [1.0] * 4,
        8,
        4,
    )
    pixels = torch.rand(1, 5, 3, 32, 32)
    latents = vae.encode(pixels)
    assert latents.shape == (1, 2, 4, 4, 4)
    assert vae.decode(latents).shape == pixels.shape
    with pytest.raises(ValueError, match="1 \\+ 4 n"):
        vae.encode(pixels[:, :4])
