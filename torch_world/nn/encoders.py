import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_world.nn.vision_transformer import VisionTransformer


class PixelPatches(nn.Module):
    """Raw pixel patches as tokens: ``(N, C, H, W)`` to ``(N, H/p * W/p, C * p * p)``, with no parameters.

    A fast, self-contained stand-in for a frozen pre-trained encoder such as DINOv2 or V-JEPA. Because the features
    are pixels, ``unpatchify`` turns predicted tokens back into images.
    """

    def __init__(self, patch_size: int, channels: int = 3):
        super().__init__()
        self.patch_size, self.dim = patch_size, channels * patch_size**2

    def forward(self, x: Tensor) -> Tensor:
        return F.pixel_unshuffle(x, self.patch_size).flatten(2).transpose(1, 2)

    def unpatchify(self, tokens: Tensor, grid: tuple[int, int]) -> Tensor:
        """Tokens ``(..., P, C * p * p)`` on a ``grid`` of patches back to images ``(..., C, H, W)``."""
        x = tokens[..., : self.dim].transpose(-1, -2).unflatten(-1, grid)
        return F.pixel_shuffle(x.flatten(0, -4), self.patch_size).unflatten(0, tokens.shape[:-2])


class CLSEncoder(nn.Module):
    """A ViT whose CLS token embeds each image: ``(N, C, H, W)`` to ``(N, embed_dim)``.

    With ``resize``, inputs of any size are resized to ``img_size`` first, as for the local crops of multi-crop
    training. Other keyword arguments go to ``VisionTransformer``.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        resize: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.img_size, self.resize, self.embed_dim = img_size, resize, embed_dim
        self.vit = VisionTransformer(
            img_size, patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, cls_token=True, **kwargs
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.resize and x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=self.img_size, mode="bilinear")
        return self.vit(x)[:, 0]


class PatchTokens(nn.Module):
    """Wraps a DINOv2 model to return its normalized patch tokens ``(N, P, D)``.

    Inputs are float images in ``[0, 1]`` or uint8 images. ``size`` resizes them first, and ``normalize`` applies the
    ImageNet mean and standard deviation that DINOv2 expects. ``PatchTokens.from_hub`` loads a released model.

    Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", TMLR 2024.
    Paper: https://arxiv.org/abs/2304.07193
    Code: https://github.com/facebookresearch/dinov2
    """

    def __init__(self, model: nn.Module, size: int | None = None, normalize: bool = False):
        super().__init__()
        self.model = model
        self.size, self.normalize = size, normalize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[:, None, None], persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[:, None, None], persistent=False)

    @classmethod
    def from_hub(cls, name: str = "dinov2_vits14", size: int | None = 224) -> "PatchTokens":
        """A frozen, normalized released DINOv2 (``dinov2_vits14``, ``_vitb14``, ``_vitl14`` or ``_vitg14``)."""
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="xFormers")
            model = torch.hub.load("facebookresearch/dinov2", name, verbose=False)
        return cls(model, size=size, normalize=True).requires_grad_(False).eval()

    @property
    def dim(self) -> int:
        return self.model.embed_dim

    def forward(self, x: Tensor) -> Tensor:
        x = x.float() / 255 if x.dtype == torch.uint8 else x
        if self.size is not None:
            x = F.interpolate(x, size=self.size, mode="bilinear", antialias=True)
        if self.normalize:
            x = (x - self.mean) / self.std
        return self.model.forward_features(x)["x_norm_patchtokens"]
