"""Video autoencoders mapping clips ``(B, T, 3, H, W)`` in ``[0, 1]`` to latents ``(B, T', C, h, w)`` and back.

Both classes expose ``latent_channels``, ``spatial_ratio`` and ``temporal_ratio``. A causal video VAE keeps the first
frame alone and packs every following ``temporal_ratio`` frames into one latent frame, so ``T' = 1 + (T - 1) / r``.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ConvVideoAE(nn.Module):
    """Small trainable per-frame convolutional autoencoder, a download-free stand-in for ``VideoVAE``.

    Each of ``depth`` stages halves the resolution, so ``spatial_ratio = 2 ** depth`` and ``temporal_ratio = 1``.
    ``forward(pixels)`` returns the reconstruction loss.
    """

    temporal_ratio = 1

    def __init__(self, latent_channels: int = 8, width: int = 32, depth: int = 2):
        super().__init__()
        self.latent_channels, self.spatial_ratio = latent_channels, 2**depth
        down, up, c = [], [], 3
        for i in range(depth):
            out = width * 2**i
            down += [nn.Conv2d(c, out, 4, stride=2, padding=1), nn.GroupNorm(8, out), nn.SiLU()]
            c = out
        down.append(nn.Conv2d(c, latent_channels, 3, padding=1))
        up.append(nn.Conv2d(latent_channels, c, 3, padding=1))
        for i in reversed(range(depth)):
            out = width * 2 ** max(i - 1, 0)
            up += [nn.GroupNorm(8, c), nn.SiLU(), nn.ConvTranspose2d(c, out, 4, stride=2, padding=1)]
            c = out
        up.append(nn.Conv2d(c, 3, 3, padding=1))
        self.encoder, self.decoder = nn.Sequential(*down), nn.Sequential(*up)

    def encode(self, pixels: Tensor) -> Tensor:
        return self.encoder(pixels.flatten(0, 1) * 2 - 1).unflatten(0, pixels.shape[:2])

    def decode(self, latents: Tensor) -> Tensor:
        return (self.decoder(latents.flatten(0, 1)).unflatten(0, latents.shape[:2]) + 1) / 2

    def forward(self, pixels: Tensor) -> dict[str, Tensor]:
        loss = F.mse_loss(self.decode(self.encode(pixels)), pixels)
        return {"loss": loss, "reconstruction_loss": loss}


class VideoVAE(nn.Module):
    """Frozen pre-trained causal video VAE from ``diffusers``, with latents normalized per channel.

    ``VideoVAE.from_pretrained()`` loads the Wan 2.1 VAE that DreamZero builds on (``spatial_ratio = 8``,
    ``temporal_ratio = 4``, 16 channels). Clips need ``1 + 4 n`` frames. Requires ``pip install torch-world[wam]``.
    """

    def __init__(self, vae: nn.Module, mean, std, spatial_ratio: int, temporal_ratio: int):
        super().__init__()
        self.vae = vae.eval().requires_grad_(False)
        self.latent_channels = len(mean)
        self.spatial_ratio, self.temporal_ratio = spatial_ratio, temporal_ratio
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32)[:, None, None])
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32)[:, None, None])

    @classmethod
    def from_pretrained(
        cls, name: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder: str = "vae", **kwargs
    ) -> "VideoVAE":
        from diffusers import AutoencoderKLWan

        vae = AutoencoderKLWan.from_pretrained(name, subfolder=subfolder, **kwargs)
        config = vae.config
        return cls(
            vae,
            config.latents_mean,
            config.latents_std,
            2 ** len(config.temperal_downsample),
            2 ** sum(config.temperal_downsample),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        return self

    @torch.no_grad()
    def encode(self, pixels: Tensor) -> Tensor:
        if (pixels.size(1) - 1) % self.temporal_ratio:
            raise ValueError(f"Clips need 1 + {self.temporal_ratio} n frames")
        x = (pixels * 2 - 1).transpose(1, 2).to(self.vae.dtype)
        latents = self.vae.encode(x).latent_dist.mode().transpose(1, 2).float()
        return (latents - self.mean) / self.std

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        x = (latents * self.std + self.mean).transpose(1, 2).to(self.vae.dtype)
        return ((self.vae.decode(x).sample.transpose(1, 2).float() + 1) / 2).clamp(0, 1)
