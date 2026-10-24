import torch
from torch import Tensor, nn

from torch_world.losses.sigreg import SIGReg
from torch_world.models.base import JEPA, WorldModel
from torch_world.nn.transformer import MLP, Transformer


class VariationalJEPA(JEPA, WorldModel):
    """Variational JEPA (VJEPA): a JEPA world model whose predictor outputs a Gaussian over the next embedding.

    The encoder and action-conditioned causal transformer follow LeWM. The predictor head returns the mean and
    log-variance of ``p(z_{t+1} | z_{<=t}, a_{<=t}) = N(mu, diag(sigma^2))``. Targets come from ``q(z | x)``, a diagonal
    Gaussian whose mean is the embedding of an EMA target encoder and whose log-variance is a learned head on it; a
    reparameterized sample of ``q`` is scored under the predictor's Gaussian:

    ``loss = nll + beta * KL(q || N(0, I)) + lamb * sigreg``, with ``nll = (log sigma^2 + (z - mu)^2 / sigma^2) / 2``

    averaged over dimensions. SIGReg keeps the online embeddings from collapsing, the role VICReg plays in the
    reference control experiments. ``predict`` returns the mean, the MAP plan the paper shows is optimal for quadratic
    costs and action-independent noise; ``predict(..., sample=True)`` draws from the Gaussian, so CEM can plan on
    sampled rollouts, and ``sample`` draws several rollouts.

    Huang, "VJEPA: Variational Joint Embedding Predictive Architectures as Probabilistic World Models", 2026.
    Paper: https://arxiv.org/abs/2601.14354
    Code: https://github.com/YongchaoHuang/VJEPA

    Args:
        encoder: module mapping images ``(N, C, H, W)`` to embeddings ``(N, embed_dim)``.
        action_dim: dimension of one action, times the frameskip when actions are stacked.
        beta: weight of the KL between the target distribution and the standard normal prior.
        logvar_range: clamp of the predicted log-variances, as the reference does.
    """

    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int,
        embed_dim: int = 192,
        history: int = 3,
        depth: int = 6,
        num_heads: int = 16,
        head_dim: int = 64,
        mlp_dim: int = 2048,
        proj_dim: int = 2048,
        dropout: float = 0.1,
        beta: float = 0.01,
        lamb: float = 0.09,
        logvar_range: tuple[float, float] = (-10.0, 10.0),
        sigreg: SIGReg | None = None,
    ):
        super().__init__()
        self.history, self.beta, self.lamb, self.logvar_range = history, beta, lamb, logvar_range
        self.encoder = nn.Sequential(encoder, MLP([embed_dim, proj_dim, embed_dim], norm=nn.BatchNorm1d))
        self.target_encoder = self.make_target_encoder(self.encoder)
        self.target_logvar = nn.Linear(embed_dim, embed_dim)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, action_dim), MLP([action_dim, 4 * embed_dim, embed_dim], act=nn.SiLU)
        )
        self.pos_embed = nn.Parameter(torch.randn(1, history, embed_dim))
        self.predictor = Transformer(
            embed_dim, depth, num_heads, conditional=True, mlp_dim=mlp_dim, head_dim=head_dim, dropout=dropout
        )
        self.head = MLP([embed_dim, proj_dim, 2 * embed_dim], norm=nn.BatchNorm1d)
        self.sigreg = sigreg or SIGReg()

    def encode(self, pixels: Tensor, encoder: nn.Module | None = None) -> Tensor:
        """``(B, T, C, H, W)`` to ``(B, T, embed_dim)``."""
        return (encoder or self.encoder)(pixels.flatten(0, 1)).unflatten(0, pixels.shape[:2])

    def distribution(self, z: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        """Mean and log-variance ``(B, T, D)`` of each next embedding, with teacher forcing."""
        B, T = z.shape[:2]
        h = self.predictor(z + self.pos_embed[:, :T], self.action_encoder(actions), is_causal=True)
        mean, logvar = self.head(h.flatten(0, 1)).unflatten(0, (B, T)).chunk(2, dim=-1)
        return mean, logvar.clamp(*self.logvar_range)

    def predict(
        self, z: Tensor, actions: Tensor, sample: bool = False, generator: torch.Generator | None = None
    ) -> Tensor:
        """The predictive mean, or a sample of the Gaussian when ``sample``."""
        mean, logvar = self.distribution(z, actions)
        if not sample:
            return mean
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        return mean + (0.5 * logvar).exp() * noise

    def sample(self, z: Tensor, actions: Tensor, num_samples: int, generator: torch.Generator | None = None) -> Tensor:
        """``num_samples`` sampled rollouts per batch element: ``(B, num_samples, T + 1, D)``."""
        z, actions = z.repeat_interleave(num_samples, 0), actions.repeat_interleave(num_samples, 0)
        return self.rollout(z, actions, sample=True, generator=generator).unflatten(0, (-1, num_samples))

    def forward(self, pixels: Tensor, actions: Tensor) -> dict[str, Tensor]:
        """``pixels`` ``(B, history + 1, C, H, W)`` and ``actions`` ``(B, history + 1, action_dim)``.

        Call ``update_target(momentum)`` after each optimizer step.
        """
        H = self.history
        z = self.encode(pixels)
        with torch.no_grad():
            target_mean = self.encode(pixels[:, 1 : H + 1], self.target_encoder)
        target_logvar = self.target_logvar(target_mean).clamp(*self.logvar_range)
        target = target_mean + (0.5 * target_logvar).exp() * torch.randn_like(target_mean)
        mean, logvar = self.distribution(z[:, :H], actions[:, :H])
        nll = 0.5 * (logvar + (target - mean).square() / logvar.exp()).mean()
        kl = 0.5 * (target_logvar.exp() + target_mean.square() - 1 - target_logvar).mean()
        sigreg = self.sigreg(z.transpose(0, 1))
        loss = nll + self.beta * kl + self.lamb * sigreg
        return {
            "loss": loss,
            "nll_loss": nll,
            "kl_loss": kl,
            "sigreg_loss": sigreg,
            "variance": logvar.detach().exp().mean(),
        }
