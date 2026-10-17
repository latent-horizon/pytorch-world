"""LeWAM's ResNet18 + spatial-keypoint visual features, using only PyTorch.

Reference architecture: MinghaoFu/lewam, models/encoders.py (MIT; see licenses/).
The projector belongs to LeWAM: use feature_dim=64, projector_norm="batch".
Inputs use ImageNet normalization, including when weights are trained from scratch.
"""

import torch
from torch import nn


class _ResidualBlock(nn.Module):
    def __init__(self, inputs, outputs, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, outputs, 3, stride, 1, bias=False)
        self.bn1 = nn.GroupNorm(16, outputs)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, 1, 1, bias=False)
        self.bn2 = nn.GroupNorm(16, outputs)
        self.downsample = (
            nn.Sequential(nn.Conv2d(inputs, outputs, 1, stride, bias=False), nn.GroupNorm(16, outputs))
            if stride != 1 or inputs != outputs
            else None
        )

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + residual)


class LeWAMEncoder(nn.Module):
    """ResNet18/GN16 trunk with 32 spatial softmax keypoints (64 features).

    Defaults match the reference's resnet18dp: 224px inputs, independent random
    202px crops during training and center crops in evaluation. No pretrained
    weights or torchvision binary extensions are required. Trunk parameter names
    match a torchvision ResNet18 with its pooling/classifier removed.
    """

    def __init__(self, crop_size: int | None = 202):
        super().__init__()
        if crop_size is not None and crop_size < 1:
            raise ValueError("crop_size must be positive or None")
        self.crop_size = crop_size
        stages = []
        inputs = 64
        for outputs, stride in ((64, 1), (128, 2), (256, 2), (512, 2)):
            stages.append(nn.Sequential(_ResidualBlock(inputs, outputs, stride), _ResidualBlock(outputs, outputs, 1)))
            inputs = outputs
        self.trunk = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            *stages,
        )
        for module in self.trunk.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        self.kp_conv = nn.Conv2d(512, 32, 1)

    def crop(self, pixels):
        n, _, h, w = pixels.shape
        size = self.crop_size
        if size is None:
            return pixels
        if min(h, w) < size:
            raise ValueError("Images must be at least crop_size in both dimensions")
        if self.training:
            top = torch.randint(h - size + 1, (n,), device=pixels.device)
            left = torch.randint(w - size + 1, (n,), device=pixels.device)
        else:
            top = torch.full((n,), (h - size) // 2, device=pixels.device, dtype=torch.long)
            left = torch.full((n,), (w - size) // 2, device=pixels.device, dtype=torch.long)
        windows = pixels.unfold(2, size, 1).unfold(3, size, 1)
        return windows[torch.arange(n, device=pixels.device), :, top, left]

    def forward(self, pixels):
        fmap = self.kp_conv(self.trunk(self.crop(pixels)))
        _, _, h, w = fmap.shape
        attention = fmap.flatten(2).softmax(-1)
        x = torch.linspace(-1, 1, w, device=fmap.device, dtype=fmap.dtype).repeat(h)
        y = torch.linspace(-1, 1, h, device=fmap.device, dtype=fmap.dtype).repeat_interleave(w)
        return torch.cat(((attention * x).sum(-1), (attention * y).sum(-1)), -1)
