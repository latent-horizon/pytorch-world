import torch
import torch.nn.functional as F
from torch import Tensor


def rollout_error(pred: Tensor, target: Tensor, p: int = 1) -> Tensor:
    """Mean ``L_p`` error per step between predicted and encoded latents ``(B, T, ...)``; returns ``(T,)``.

    Open-loop prediction error grows with the horizon; its curve compares world models on the same encoder. V-JEPA 2-AC
    trains and evaluates with ``p = 1``; ``p = 2`` gives the mean squared error of DINO-WM and LeWM. Spikes of the
    one-step error measure surprise, as in LeWM's violation-of-expectation tests.
    """
    diff = (pred - target).flatten(2).abs()
    return diff.pow(p).mean((0, 2))


def psnr(pred: Tensor, target: Tensor, data_range: float = 1.0) -> Tensor:
    """Peak signal-to-noise ratio in dB of images ``(..., C, H, W)``, one value per image."""
    mse = (pred - target).square().flatten(-3).mean(-1).clamp_min(1e-12)
    return 10 * torch.log10(data_range**2 / mse)


def ssim(pred: Tensor, target: Tensor, data_range: float = 1.0, window: int = 11, sigma: float = 1.5) -> Tensor:
    """Structural similarity of images ``(..., C, H, W)`` with a Gaussian window, one value per image, as DINO-WM reports
    for decoded predictions.

    Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity", IEEE TIP 2004.
    Paper: https://doi.org/10.1109/TIP.2003.819861
    """
    shape = pred.shape[:-3]
    x, y = pred.reshape(-1, *pred.shape[-3:]).float(), target.reshape(-1, *target.shape[-3:]).float()
    c = x.size(1)
    g = torch.exp(-(torch.arange(window, device=x.device) - window // 2).float().square() / (2 * sigma**2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None]).expand(c, 1, window, window)
    blur = lambda t: F.conv2d(t, kernel, groups=c)
    mx, my = blur(x), blur(y)
    vx, vy, cov = blur(x * x) - mx.square(), blur(y * y) - my.square(), blur(x * y) - mx * my
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = (2 * mx * my + c1) * (2 * cov + c2) / ((mx.square() + my.square() + c1) * (vx + vy + c2))
    return s.flatten(1).mean(1).view(shape)


def chamfer_distance(x: Tensor, y: Tensor) -> Tensor:
    """Symmetric Chamfer distance between point sets ``(B, N, D)`` and ``(B, M, D)``: the mean squared distance from
    each point to its nearest neighbor in the other set, summed over both directions. DINO-WM scores its deformable
    tasks (rope, granular) with it."""
    d = torch.cdist(x.float(), y.float()).square()
    return d.min(2).values.mean(1) + d.min(1).values.mean(1)
