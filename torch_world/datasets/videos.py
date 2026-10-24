import torch
import torch.nn.functional as F
from torch import Tensor


def read_video(path: str, size: tuple[int, int] | None = None, device=None) -> Tensor:
    """Every frame of a video file as ``(F, 3, H, W)`` in ``[0, 1]``, resized to ``size`` when given. Needs ``av``."""
    import av
    import numpy as np

    with av.open(str(path)) as container:
        frames = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
    frames = torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float().div(255)
    return frames if size is None else F.interpolate(frames, size=size, mode="bilinear", antialias=True)


def random_clips(
    frames: Tensor, n: int, length: int, size: int, stride: int = 1, generator: torch.Generator | None = None
) -> Tensor:
    """``n`` clips ``(n, 3, length, size, size)`` of ``frames`` ``(F, 3, H, W)``: each a fixed square window, at a
    random place and time, over ``length`` frames ``stride`` apart, as a fixed camera films them."""
    F_, _, H, W = frames.shape
    start = torch.randint(F_ - (length - 1) * stride, (n,), generator=generator)
    y, x = (
        torch.randint(H - size + 1, (n,), generator=generator),
        torch.randint(W - size + 1, (n,), generator=generator),
    )
    time = start[:, None] + stride * torch.arange(length)
    rows, cols = y[:, None] + torch.arange(size), x[:, None] + torch.arange(size)
    clips = frames[time[:, :, None, None], :, rows[:, None, :, None], cols[:, None, None, :]]  # (n, T, S, S, 3)
    return clips.permute(0, 4, 1, 2, 3)
