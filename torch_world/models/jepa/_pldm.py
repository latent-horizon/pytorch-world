import torch
from torch import Tensor, nn

from torch_world.models.base import WorldModel
from torch_world.nn.transformer import MLP


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.body(x)


class PLDMEncoder(nn.Sequential):
    """A small IMPALA-style encoder: ``(N, C, H, W)`` to ``(N, latent_dim)``.

    Each stage is a convolution, a stride-2 max pool and two residual blocks, as in the IMPALA encoder that PLDM
    takes from OGBench, followed by a linear layer to ``latent_dim``.
    """

    def __init__(
        self, img_size: int, latent_dim: int = 128, in_channels: int = 3, channels: tuple[int, ...] = (16, 32, 32)
    ):
        layers, c = [], in_channels
        for out in channels:
            layers += [nn.Conv2d(c, out, 3, padding=1), nn.MaxPool2d(3, 2, 1), ResidualBlock(out), ResidualBlock(out)]
            c, img_size = out, (img_size + 1) // 2
        super().__init__(*layers, nn.ReLU(), nn.Flatten(), nn.Linear(c * img_size**2, latent_dim))


class PLDM(WorldModel):
    """PLDM: a JEPA latent dynamics model trained reward-free, then used to plan toward goal images.

    Each frame is encoded to one latent, normalized by a LayerNorm. A GRU whose hidden state is the latent predicts
    the next latent from the current one and the action, through the same LayerNorm. During training it is unrolled
    open loop from the first frame over the whole window, and five terms keep the latents informative:

    ``loss = pred + var_coeff * var + cov_coeff * cov + smooth_coeff * smooth + idm_coeff * idm``

    ``pred`` is the MSE of the multi-step predictions, ``var`` and ``cov`` are VICReg's variance hinge and covariance
    penalty over the batch at every step, ``smooth`` pulls consecutive latents together and ``idm`` is an inverse
    dynamics MLP regressing ``a_t`` from ``(z_t, z_{t+1})``. Planning minimizes the latent distance to the goal along
    the imagined path.

    Sobal et al., "Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models", 2025.
    Paper: https://arxiv.org/abs/2502.14819
    Code: https://github.com/vladisai/PLDM

    Args:
        encoder: module mapping images ``(N, C, H, W)`` to latents ``(N, latent_dim)``, such as ``PLDMEncoder``.
        action_dim: dimension of one action.
        var_coeff, cov_coeff, smooth_coeff, idm_coeff: loss weights; the defaults are the reference's Two-Rooms ones.
    """

    history = 1

    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int,
        latent_dim: int = 128,
        idm_dim: int = 512,
        var_coeff: float = 4.0,
        cov_coeff: float = 6.9,
        smooth_coeff: float = 0.75,
        idm_coeff: float = 1.0,
        std_margin: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.norm = nn.LayerNorm(latent_dim)
        self.predictor = nn.GRUCell(action_dim, latent_dim)
        self.idm = MLP([2 * latent_dim, idm_dim, action_dim], act=nn.ReLU)
        self.var_coeff, self.cov_coeff, self.std_margin = var_coeff, cov_coeff, std_margin
        self.smooth_coeff, self.idm_coeff = smooth_coeff, idm_coeff

    def encode(self, pixels: Tensor) -> Tensor:
        """``(B, T, C, H, W)`` to ``(B, T, latent_dim)``."""
        return self.norm(self.encoder(pixels.flatten(0, 1))).unflatten(0, pixels.shape[:2])

    def step(self, z: Tensor, action: Tensor) -> Tensor:
        """One step of the dynamics: ``(N, latent_dim)`` and ``(N, action_dim)`` to the next latent."""
        return self.norm(self.predictor(action, z))

    def predict(self, z: Tensor, actions: Tensor) -> Tensor:
        return self.step(z.flatten(0, 1), actions.flatten(0, 1)).unflatten(0, z.shape[:2])

    def rollout(self, z: Tensor, actions: Tensor) -> Tensor:
        """Unroll from the last context latent ``z[:, -1]`` with ``actions[:, T0 - 1:]``; returns ``T + 1`` latents."""
        out = list(z.unbind(1))
        for t in range(z.size(1) - 1, actions.size(1)):
            out.append(self.step(out[-1], actions[:, t]))
        return torch.stack(out, 1)

    def forward(self, pixels: Tensor, actions: Tensor) -> dict[str, Tensor]:
        """``pixels`` ``(B, T + 1, C, H, W)`` and ``actions`` ``(B, T + 1, action_dim)``; the last action is unused."""
        z = self.encode(pixels)
        T, D = z.size(1) - 1, z.size(2)
        pred = self.rollout(z[:, :1], actions[:, :T])
        pred_loss = (pred[:, 1:] - z[:, 1:]).square().mean()

        zt = z.transpose(0, 1)
        zt = zt - zt.mean(1, keepdim=True)
        var_loss = (self.std_margin - (zt.var(1) + 1e-4).sqrt()).relu().mean()
        cov = zt.transpose(1, 2) @ zt / (zt.size(1) - 1)
        off_diagonal = cov.square().sum((1, 2)) - cov.diagonal(dim1=1, dim2=2).square().sum(1)
        cov_loss = (off_diagonal / (D * (D - 1))).mean()

        smooth_loss = (z[:, 1:] - z[:, :-1]).square().mean()
        idm_loss = (self.idm(torch.cat([z[:, :-1], z[:, 1:]], -1)) - actions[:, :T]).square().mean()

        loss = (
            pred_loss
            + self.var_coeff * var_loss
            + self.cov_coeff * cov_loss
            + self.smooth_coeff * smooth_loss
            + self.idm_coeff * idm_loss
        )
        return {
            "loss": loss,
            "pred_loss": pred_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
            "smooth_loss": smooth_loss,
            "idm_loss": idm_loss,
        }
