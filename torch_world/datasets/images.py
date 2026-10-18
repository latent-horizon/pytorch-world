import torch
import torch.nn.functional as F
from torch import Tensor


def resize_images(images, size: int, chunk: int = 64) -> Tensor:
    """RGB images ``(..., H, W, 3)`` uint8 to ``(..., 3, size, size)`` uint8 by area averaging.

    The whole frame is resized, so 640 x 480 cameras are squeezed to a square rather than cropped. Loaders and
    policy servers share this function, so a policy sees the same pixels in training and evaluation.
    """
    images = torch.as_tensor(images)
    lead, frames = images.shape[:-3], images.reshape(-1, *images.shape[-3:]).movedim(-1, 1)
    out = [
        F.interpolate(frames[i : i + chunk].float(), (size, size), mode="area").round().to(torch.uint8)
        for i in range(0, len(frames), chunk)
    ]
    return torch.cat(out).reshape(*lead, 3, size, size)
