import math

from torch import Tensor, nn

from torch_world.nn.transformer import MLP


def flat(z: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Latents ``(..., *shape)`` to vectors ``(..., prod(shape))``."""
    return z.reshape(*z.shape[: z.dim() - len(shape)], math.prod(shape))


class FailureMargin(nn.Module):
    """A failure classifier on the latents of a frozen world model, read as a signed safety margin ``l(z)``.

    An MLP scores each latent; ``margin`` squashes the score with ``tanh`` so that ``l(z)`` lies in ``(-1, 1)``, is
    positive on safe latents and negative on failures, and the failure set is ``{z : l(z) < 0}``. The score is
    trained with two hinges that push safe latents above ``delta`` and failures below ``-delta``, each averaged over
    its own class so that rare failures weigh as much as the safe majority:

    ``loss = mean_safe relu(delta - s(z)) + mean_failure relu(delta + s(z))``

    Nakamura, Peters and Bajcsy, "Generalizing Safety Beyond Collision-Avoidance via Latent-Space Reachability Analysis", 2025.
    Paper: https://arxiv.org/abs/2502.00935
    Code: https://github.com/CMU-IntentLab/latent-safety

    Args:
        latent_shape: shape of one latent, for example ``128`` or ``(196, 384)``; latents are flattened.
        hidden: hidden widths of the MLP.
        delta: hinge margin; the paper leaves it unreported, the reference code uses 0.75.
    """

    def __init__(self, latent_shape: int | tuple[int, ...], hidden: tuple[int, ...] = (256, 256), delta: float = 0.75):
        super().__init__()
        self.shape = (latent_shape,) if isinstance(latent_shape, int) else tuple(latent_shape)
        self.net = MLP([math.prod(self.shape), *hidden, 1], act=nn.ReLU)
        self.delta = delta

    def score(self, z: Tensor) -> Tensor:
        """Raw score ``(...,)`` of latents ``(..., *latent_shape)``, the quantity the hinges train."""
        return self.net(flat(z, self.shape))[..., 0]

    def margin(self, z: Tensor) -> Tensor:
        """Safety margin ``l(z) = tanh(score(z))`` ``(...,)``; negative means failure."""
        return self.score(z).tanh()

    def forward(self, z: Tensor, failure: Tensor) -> dict[str, Tensor]:
        """``z`` ``(N, *latent_shape)`` and ``failure`` labels ``(N,)`` (true or 1 for failures)."""
        s, failure = self.score(z), failure.bool().reshape(-1)
        s = s.reshape(-1)
        safe_loss = (self.delta - s[~failure]).relu().mean() if (~failure).any() else s.new_zeros(())
        failure_loss = (self.delta + s[failure]).relu().mean() if failure.any() else s.new_zeros(())
        accuracy = ((s >= 0) != failure).float().mean()
        return {
            "loss": safe_loss + failure_loss,
            "safe_loss": safe_loss,
            "failure_loss": failure_loss,
            "accuracy": accuracy,
        }
