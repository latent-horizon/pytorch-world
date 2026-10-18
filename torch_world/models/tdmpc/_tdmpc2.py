import copy
import math
from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call

from torch_world.models.base import WorldModel
from torch_world.utils.ema import ema_update


class SimNorm(nn.Module):
    """Simplicial normalization: a softmax over each group of ``dim`` features.

    Lavoie et al., "Simplicial Embeddings in Self-Supervised Learning and Downstream Classification", ICLR 2023.
    Paper: https://arxiv.org/abs/2204.00616
    """

    def __init__(self, dim: int = 8):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.unflatten(-1, (-1, self.dim)).softmax(-1).flatten(-2)


class ShiftAug(nn.Module):
    """Random shift of square images by up to ``pad`` pixels, with replicated borders (DrQ-v2).

    As in TD-MPC2's reference code, the shift is applied in eval mode too.
    """

    def __init__(self, pad: int = 3):
        super().__init__()
        self.pad = pad

    def forward(self, x: Tensor) -> Tensor:
        n, _, h, w = x.shape
        if h != w:
            raise ValueError(f"ShiftAug expects square images, got {h}x{w}")
        x = F.pad(x, (self.pad,) * 4, mode="replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange[None].expand(h, -1)
        base = torch.stack([arange, arange.T], dim=-1)[None].expand(n, -1, -1, -1)
        shift = torch.randint(0, 2 * self.pad + 1, (n, 1, 1, 2), device=x.device).to(x.dtype)
        return F.grid_sample(x, base + shift * 2.0 / (h + 2 * self.pad), padding_mode="zeros", align_corners=False)


class _Center(nn.Module):
    """Images in ``[0, 1]`` (or uint8) to ``[-0.5, 0.5]``."""

    def forward(self, x: Tensor) -> Tensor:
        return (x.float() / 255 if x.dtype == torch.uint8 else x) - 0.5


def mlp(dims: list[int], act: nn.Module | None = None, dropout: float = 0.0) -> nn.Sequential:
    """TD-MPC2's MLP through ``dims``: Linear, LayerNorm and Mish per hidden layer, with dropout after the first
    linear only. The output layer is a plain Linear, or Linear, LayerNorm and ``act`` when ``act`` is given."""

    def normed(a: int, b: int, fn: nn.Module, p: float) -> nn.Sequential:
        return nn.Sequential(nn.Linear(a, b), *([nn.Dropout(p)] if p else []), nn.LayerNorm(b), fn)

    layers = [normed(a, b, nn.Mish(), dropout if i == 0 else 0.0) for i, (a, b) in enumerate(pairwise(dims[:-1]))]
    layers.append(normed(dims[-2], dims[-1], act, 0.0) if act else nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


def _init(module: nn.Module) -> nn.Module:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
    return module


def state_encoder(
    obs_dim: int, latent_dim: int = 512, hidden_dim: int = 256, num_layers: int = 2, simnorm_dim: int = 8
) -> nn.Sequential:
    """TD-MPC2's encoder for state vectors ``(N, obs_dim)``: an MLP to ``(N, latent_dim)`` ending in ``SimNorm``."""
    return _init(mlp([obs_dim, *[hidden_dim] * max(num_layers - 1, 1), latent_dim], act=SimNorm(simnorm_dim)))


def pixel_encoder(in_channels: int = 3, num_channels: int = 32, simnorm_dim: int = 8) -> nn.Sequential:
    """TD-MPC2's encoder for 64x64 images ``(N, C, 64, 64)`` in ``[0, 1]`` or uint8: random shift, four convolutions
    and ``SimNorm``. The latent has ``num_channels * 16`` features, 512 by default."""
    return nn.Sequential(
        ShiftAug(),
        _Center(),
        nn.Conv2d(in_channels, num_channels, 7, stride=2),
        nn.ReLU(),
        nn.Conv2d(num_channels, num_channels, 5, stride=2),
        nn.ReLU(),
        nn.Conv2d(num_channels, num_channels, 3, stride=2),
        nn.ReLU(),
        nn.Conv2d(num_channels, num_channels, 3, stride=1),
        nn.Flatten(),
        SimNorm(simnorm_dim),
    )


def symlog(x: Tensor) -> Tensor:
    return x.sign() * x.abs().log1p()


def symexp(x: Tensor) -> Tensor:
    return x.sign() * x.abs().expm1()


class TDMPC2(WorldModel):
    """TD-MPC2: a decoder-free latent world model trained with temporal-difference learning, planned with MPPI.

    An encoder maps observations to ``SimNorm`` latents. From the first latent, a deterministic MLP dynamics model
    unrolls the batch's actions; every imagined latent is pulled toward the encoding of the real next observation
    (consistency), and predicts the reward and, through an ensemble of ``num_q`` Q-functions, the TD target
    ``r + discount * Q_target(z', pi(z'))``. Rewards and values are regressed as discrete distributions: cross-entropy
    against two-hot targets over ``num_bins`` bins in symlog space. All terms are weighted by ``rho ** t``.

    A tanh-Gaussian policy prior maximizes the scale-normalized Q-value plus an entropy bonus, on detached latents and
    through Q-functions it cannot update; it proposes trajectories to the planner and bootstraps returns. Plan with
    ``MPPI``, which scores action sequences with ``estimate_value``. Actions are in ``[-1, 1]``.

    ``forward`` returns the world-model loss plus the policy loss: their gradients reach disjoint parameters, so one
    backward pass matches the reference's two updates, except that the policy loss sees the Q-functions before, not
    after, this step's update. Train with ``param_groups``, clip with ``clip_grad_norm`` and call ``update_target``
    after every optimizer step. The reference's multi-task embeddings are not included.

    Hansen et al., "TD-MPC2: Scalable, Robust World Models for Continuous Control", ICLR 2024.
    Paper: https://arxiv.org/abs/2310.16828
    Code: https://github.com/nicklashansen/tdmpc2

    Args:
        encoder: module mapping observations ``(N, ...)`` to ``SimNorm`` latents ``(N, latent_dim)``, such as
            ``state_encoder`` or ``pixel_encoder``.
        action_dim: dimension of one action.
        discount: reward discount; ``TDMPC2.episode_discount`` gives the reference's choice for an episode length.
        episodic: learn a termination head, used in the TD targets and to stop imagined returns.
    """

    history = 1

    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int,
        latent_dim: int = 512,
        mlp_dim: int = 512,
        num_q: int = 5,
        dropout: float = 0.01,
        simnorm_dim: int = 8,
        num_bins: int = 101,
        vmin: float = -10.0,
        vmax: float = 10.0,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        discount: float = 0.99,
        rho: float = 0.5,
        consistency_coef: float = 20.0,
        reward_coef: float = 0.1,
        value_coef: float = 0.1,
        termination_coef: float = 1.0,
        entropy_coef: float = 1e-4,
        scale_momentum: float = 0.99,
        episodic: bool = False,
    ):
        super().__init__()
        self.action_dim, self.num_q, self.discount, self.rho = action_dim, num_q, discount, rho
        self.vmin, self.vmax, self.bin_size = vmin, vmax, (vmax - vmin) / (num_bins - 1)
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        self.consistency_coef, self.reward_coef, self.value_coef = consistency_coef, reward_coef, value_coef
        self.termination_coef, self.entropy_coef, self.scale_momentum = termination_coef, entropy_coef, scale_momentum
        za, hidden = latent_dim + action_dim, [mlp_dim, mlp_dim]
        self.encoder = encoder
        self.dynamics = _init(mlp([za, *hidden, latent_dim], act=SimNorm(simnorm_dim)))
        self.reward = _init(mlp([za, *hidden, num_bins]))
        self.termination = _init(mlp([latent_dim, *hidden, 1])) if episodic else None
        self.policy = _init(mlp([latent_dim, *hidden, 2 * action_dim]))
        self.qs = nn.ModuleList(_init(mlp([za, *hidden, num_bins], dropout=dropout)) for _ in range(num_q))
        for head in [self.reward, *self.qs]:
            nn.init.zeros_(head[-1].weight)
        self.target_qs = copy.deepcopy(self.qs).requires_grad_(False).eval()
        self.register_buffer("bins", torch.linspace(vmin, vmax, num_bins))
        self.register_buffer("q_scale", torch.ones(()))

    @staticmethod
    def episode_discount(episode_length: int, denom: float = 5, low: float = 0.95, high: float = 0.995) -> float:
        """The reference's discount for fixed-length episodes: ``(L / denom - 1) / (L / denom)``, clipped."""
        frac = episode_length / denom
        return min(max((frac - 1) / frac, low), high)

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_qs.eval()
        return self

    def param_groups(self, lr: float = 3e-4, encoder_lr_scale: float = 0.3) -> list[dict]:
        """Adam groups as in the reference: the encoder at ``encoder_lr_scale * lr``, the policy with ``eps=1e-5``."""
        encoder, policy = list(self.encoder.parameters()), list(self.policy.parameters())
        skip = {id(p) for p in encoder + policy}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in skip]
        return [
            {"params": encoder, "lr": lr * encoder_lr_scale},
            {"params": rest, "lr": lr},
            {"params": policy, "lr": lr, "eps": 1e-5},
        ]

    def clip_grad_norm(self, max_norm: float = 20.0) -> tuple[Tensor, Tensor]:
        """Clip the world model's and the policy's gradients separately, as the reference does; returns both norms."""
        policy = {id(p) for p in self.policy.parameters()}
        world = [p for p in self.parameters() if p.requires_grad and id(p) not in policy]
        return nn.utils.clip_grad_norm_(world, max_norm), nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm)

    def update_target(self, momentum: float = 0.99):
        """Polyak update of the target Q-functions; the reference's ``tau`` is ``1 - momentum``."""
        ema_update(self.target_qs, self.qs, momentum)

    def two_hot(self, x: Tensor) -> Tensor:
        """Scalars ``(...)`` to soft two-hot targets ``(..., num_bins)`` over bins evenly spaced in symlog space."""
        pos = (symlog(x).clamp(self.vmin, self.vmax) - self.vmin) / self.bin_size
        idx = pos.floor()
        offset = (pos - idx)[..., None]
        idx = idx.long()[..., None]
        out = x.new_zeros(*x.shape, self.bins.numel()).scatter(-1, idx, 1 - offset)
        return out.scatter(-1, (idx + 1) % self.bins.numel(), offset)

    def two_hot_inv(self, logits: Tensor) -> Tensor:
        """Logits ``(..., num_bins)`` to the scalars ``(...)`` of their expected bin."""
        return symexp((logits.softmax(-1) * self.bins).sum(-1))

    def soft_ce(self, logits: Tensor, target: Tensor) -> Tensor:
        """Cross-entropy ``(...)`` of logits ``(..., num_bins)`` against the two-hot encoding of ``target`` ``(...)``."""
        return -(self.two_hot(target) * logits.log_softmax(-1)).sum(-1)

    def encode(self, obs: Tensor) -> Tensor:
        """``(B, T, ...)`` to ``(B, T, latent_dim)``."""
        return self.encoder(obs.flatten(0, 1)).unflatten(0, obs.shape[:2])

    def predict(self, z: Tensor, actions: Tensor) -> Tensor:
        """Next latents from latents ``(..., latent_dim)`` and actions ``(..., action_dim)``."""
        return self.dynamics(torch.cat([z, actions], dim=-1))

    def q(self, z: Tensor, actions: Tensor, reduce: str = "all", target: bool = False, detach: bool = False) -> Tensor:
        """Q-values of latents ``(..., latent_dim)`` and actions ``(..., action_dim)``.

        ``"all"`` returns the logits ``(num_q, ..., num_bins)`` of every Q-function; ``"min"`` and ``"mean"`` return
        the min or mean ``(...)`` of two Q-functions drawn at random. ``target`` uses the target Q-functions;
        ``detach`` lets gradients reach the actions and latents but not the Q-functions.
        """
        x = torch.cat([z, actions], dim=-1)
        nets = self.target_qs if target else self.qs
        if detach:
            out = [functional_call(q, {k: v.detach() for k, v in q.named_parameters()}, (x,)) for q in nets]
        else:
            out = [q(x) for q in nets]
        logits = torch.stack(out)
        if reduce == "all":
            return logits
        values = self.two_hot_inv(logits[torch.randperm(self.num_q, device=x.device)[:2]])
        if reduce == "min":
            return values.min(0).values
        if reduce == "mean":
            return values.mean(0)
        raise ValueError(f"reduce must be 'all', 'min' or 'mean', not {reduce!r}")

    def pi(self, z: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Sample tanh-squashed actions ``(..., action_dim)`` from the policy prior.

        The info dict holds the squashed ``"mean"``, the ``"log_std"``, the ``"entropy"`` of the squashed sample and
        the reference's ``"scaled_entropy"``: the Gaussian's, before squashing, times ``action_dim``.
        """
        mean, log_std = self.policy(z).chunk(2, dim=-1)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std.tanh() + 1)
        eps = torch.randn_like(mean)
        log_prob = (-0.5 * eps.square() - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        action = (mean + eps * log_std.exp()).tanh()
        squashed = log_prob - torch.log(F.relu(1 - action.square()) + 1e-6).sum(-1)
        info = {
            "mean": mean.tanh(),
            "log_std": log_std,
            "entropy": -squashed,
            "scaled_entropy": -log_prob * eps.size(-1),
        }
        return action, info

    def estimate_value(self, z: Tensor, actions: Tensor) -> Tensor:
        """Discounted return ``(N,)`` of action sequences ``(N, H, action_dim)`` from latents ``(N, latent_dim)``:
        predicted rewards, then the mean of two Q-values for a policy action at the last latent."""
        ret, discount, alive = 0.0, 1.0, z.new_ones(z.size(0))
        for a in actions.unbind(1):
            reward = self.two_hot_inv(self.reward(torch.cat([z, a], dim=-1)))
            z = self.predict(z, a)
            ret = ret + discount * alive * reward
            discount = discount * self.discount
            if self.termination is not None:
                alive = alive * (self.termination(z).squeeze(-1) <= 0)
        return ret + discount * alive * self.q(z, self.pi(z)[0], "mean")

    def forward(
        self, obs: Tensor, actions: Tensor, rewards: Tensor, terminated: Tensor | None = None
    ) -> dict[str, Tensor]:
        """``obs`` ``(B, H + 1, ...)``, ``actions`` ``(B, H, action_dim)``, ``rewards`` ``(B, H)`` and, when
        episodic, ``terminated`` ``(B, H)``: whether ``obs[:, t + 1]`` ends the episode. ``actions[:, t]`` is applied
        at ``obs[:, t]`` and earns ``rewards[:, t]``."""
        H = actions.size(1)
        terminated = torch.zeros_like(rewards) if terminated is None else terminated.to(rewards.dtype)
        with torch.no_grad():
            next_z = self.encode(obs[:, 1:])
            td_target = rewards + self.discount * (1 - terminated) * self.q(
                next_z, self.pi(next_z)[0], "min", target=True
            )

        zs = [self.encode(obs[:, :1])[:, 0]]
        for t in range(H):
            zs.append(self.predict(zs[-1], actions[:, t]))
        zs = torch.stack(zs, dim=1)
        rho = self.rho ** torch.arange(H + 1, device=zs.device, dtype=zs.dtype)
        x = torch.cat([zs[:, :-1], actions], dim=-1)
        consistency = ((zs[:, 1:] - next_z).square().mean((0, 2)) * rho[:H]).sum() / H
        reward = (self.soft_ce(self.reward(x), rewards).mean(0) * rho[:H]).sum() / H
        value = (self.soft_ce(self.q(zs[:, :-1], actions), td_target).mean((0, 1)) * rho[:H]).sum() / H
        loss = self.consistency_coef * consistency + self.reward_coef * reward + self.value_coef * value
        out = {"consistency_loss": consistency, "reward_loss": reward, "value_loss": value}
        if self.termination is not None:
            out["termination_loss"] = F.binary_cross_entropy_with_logits(
                self.termination(zs[:, 1:]).squeeze(-1), terminated
            )
            loss = loss + self.termination_coef * out["termination_loss"]

        z = zs.detach()
        action, info = self.pi(z)
        q = self.q(z, action, "mean", detach=True)
        if self.training:
            low, high = torch.quantile(q[:, 0].detach(), q.new_tensor([0.05, 0.95]))
            self.q_scale.lerp_((high - low).clamp(min=1.0), 1 - self.scale_momentum)
        pi = (-(self.entropy_coef * info["scaled_entropy"] + q / self.q_scale).mean(0) * rho).mean()
        return {"loss": loss + pi, **out, "pi_loss": pi}
