import torch
from torch import Tensor, nn

from torch_world.losses.sigreg import SIGReg
from torch_world.models.base import WorldModel
from torch_world.nn.flow import euler_sample, interpolate, sample_time, time_embedding, velocity_target
from torch_world.nn.transformer import MLP, Transformer


class FlowJEPA(WorldModel):
    """Flow-JEPA (F-JEPA): a LeWM-style JEPA world model whose predictor samples whole future trajectories by flow
    matching.

    Each frame is encoded to one embedding (the encoder's CLS token) and projected, as in LeWM. From the current
    embedding ``z_t`` and ``num_preds`` actions, a transformer predicts the velocity that transports Gaussian source
    latents ``Z_0`` to the future embeddings ``Z = [z_{t+1}, ..., z_{t+num_preds}]``, all at once. The flow time
    ``tau`` conditions every block through adaLN-zero; future token ``i`` sees ``z_t``, the actions up to ``a_{t+i}``
    and the other future tokens (all of them, or the earlier ones when ``causal``). Training regresses the straight
    path's velocity, end to end with SIGReg on the embeddings:

    ``loss = || v(Z_tau, tau, z_t, a) - (Z - Z_0) ||^2 + lamb * sigreg``, with ``Z_tau = (1 - tau) Z_0 + tau Z``.

    ``Z_0`` is ``source_std * N(0, I)``, centered on ``z_t`` when ``source="current"``. A prediction integrates the flow
    from fresh noise in ``steps`` Euler steps, so ``predict`` and ``rollout`` draw one sample of the future, and
    ``sample`` draws several. Planning scores each candidate by one sampled rollout, as the reference does.

    Huo, Luo and Song, "Flow-JEPA: Robust Latent Dynamics for JEPA World Models via Flow Matching", 2026.
    Paper: https://arxiv.org/abs/2608.29029
    Code: https://github.com/HuoYanchen/Flow-JEPA

    Args:
        encoder: module mapping images ``(N, C, H, W)`` to embeddings ``(N, embed_dim)`` (ViT-tiny CLS in the paper).
        action_dim: dimension of one action, times the frameskip when actions are stacked.
        num_preds: future embeddings predicted jointly in training (``h = 5`` in the paper).
    """

    history = 1

    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int,
        embed_dim: int = 192,
        num_preds: int = 5,
        depth: int = 6,
        num_heads: int = 16,
        head_dim: int = 64,
        mlp_dim: int = 2048,
        proj_dim: int = 2048,
        dropout: float = 0.1,
        lamb: float = 0.1,
        steps: int = 8,
        source: str = "noise",
        source_std: float = 1.0,
        causal: bool = False,
        sigreg: SIGReg | None = None,
    ):
        super().__init__()
        if source not in ("noise", "current"):
            raise ValueError(f"source must be 'noise' or 'current', not {source!r}")
        self.embed_dim, self.num_preds, self.steps, self.lamb = embed_dim, num_preds, steps, lamb
        self.source, self.source_std, self.causal = source, source_std, causal
        self.encoder = encoder
        self.projector = MLP([embed_dim, proj_dim, embed_dim], norm=nn.BatchNorm1d)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, action_dim), MLP([action_dim, 4 * embed_dim, embed_dim], act=nn.SiLU)
        )
        self.time_mlp = MLP([embed_dim, embed_dim, embed_dim], act=nn.SiLU)
        self.predictor = Transformer(
            embed_dim, depth, num_heads, conditional=True, mlp_dim=mlp_dim, head_dim=head_dim, dropout=dropout
        )
        self.out_proj = MLP([embed_dim, 4 * embed_dim, embed_dim], norm=nn.RMSNorm)
        self.sigreg = sigreg or SIGReg()

    def encode(self, pixels: Tensor) -> Tensor:
        """``(B, T, C, H, W)`` to ``(B, T, embed_dim)``."""
        return self.projector(self.encoder(pixels.flatten(0, 1))).unflatten(0, pixels.shape[:2])

    def mask(self, P: int, device) -> Tensor:
        """Boolean attention mask over ``[z_t, a_t .. a_{t+P-1}, x_1 .. x_P]``; ``True`` attends."""
        i = torch.arange(1 + 2 * P, device=device)
        step = torch.where(i <= P, i, i - P)  # z_t is step 0; a_{t+k} and x_{k+1} are step k + 1
        future = i > P
        mask = step[None, :] <= step[:, None]
        return torch.where(future[None, :], future[:, None] & (mask if self.causal else True), mask)

    def velocity(self, x: Tensor, tau: Tensor, z: Tensor, actions: Tensor) -> Tensor:
        """Velocity ``(B, P, D)`` at noisy futures ``x`` and flow times ``tau`` ``(B,)``, from ``z`` ``(B, 1, D)`` and
        encoded actions ``(B, P, D)``."""
        P = x.size(1)
        pos = time_embedding(torch.arange(1, P + 1, device=x.device, dtype=x.dtype), self.embed_dim).to(x.dtype)
        tokens = torch.cat([z, actions + pos, x + pos], dim=1)
        c = self.time_mlp(time_embedding(tau, self.embed_dim).to(x.dtype))[:, None]
        return self.out_proj(self.predictor(tokens, c, attn_mask=self.mask(P, x.device))[:, -P:])

    def source_latents(self, z: Tensor, noise: Tensor) -> Tensor:
        return self.source_std * noise + (z if self.source == "current" else 0)

    def predict(
        self,
        z: Tensor,
        actions: Tensor,
        noise: Tensor | None = None,
        steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """One sample of the ``T`` embeddings that ``actions`` ``(B, T, action_dim)`` lead to from ``z[:, 0]``.

        Later context embeddings are not used (``history = 1``); ``noise`` ``(B, T, D)`` fixes the source sample.
        """
        z0, a = z[:, :1], self.action_encoder(actions)
        if noise is None:
            shape = (z.size(0), actions.size(1), z.size(-1))
            noise = torch.randn(shape, device=z.device, dtype=z.dtype, generator=generator)
        return euler_sample(
            lambda x, t: self.velocity(x, t, z0, a), self.source_latents(z0, noise), steps or self.steps
        )

    def rollout(self, z: Tensor, actions: Tensor, **kwargs) -> Tensor:
        """Sample the future of the last context embedding jointly; returns ``T + 1`` embeddings like ``WorldModel``."""
        return torch.cat([z, self.predict(z[:, -1:], actions[:, z.size(1) - 1 :], **kwargs)], dim=1)

    def sample(
        self, z: Tensor, actions: Tensor, num_samples: int, generator: torch.Generator | None = None, **kwargs
    ) -> Tensor:
        """``num_samples`` rollouts per batch element: ``(B, num_samples, T + 1, D)``."""
        z, actions = z.repeat_interleave(num_samples, 0), actions.repeat_interleave(num_samples, 0)
        return self.rollout(z, actions, generator=generator, **kwargs).unflatten(0, (-1, num_samples))

    def forward(self, pixels: Tensor, actions: Tensor) -> dict[str, Tensor]:
        """``pixels`` ``(B, num_preds + 1, C, H, W)`` and ``actions`` ``(B, num_preds + 1, action_dim)``."""
        z, P = self.encode(pixels), self.num_preds
        target = z[:, 1 : P + 1]
        source = self.source_latents(z[:, :1], torch.randn_like(target))
        tau = sample_time(z.size(0), device=z.device, dtype=z.dtype)
        v = self.velocity(interpolate(source, target, tau), tau, z[:, :1], self.action_encoder(actions[:, :P]))
        flow = (v - velocity_target(source, target)).square().mean()
        sigreg = self.sigreg(z.transpose(0, 1))
        return {"loss": flow + self.lamb * sigreg, "flow_loss": flow, "sigreg_loss": sigreg}
