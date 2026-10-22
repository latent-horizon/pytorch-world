"""LeWAM's regression-dynamics / flow-action variant, with a pluggable encoder.

Architecture reference: https://github.com/MinghaoFu/lewam (MIT).
The reference license is included in licenses/lewam-MIT.txt.
This implementation supports one view and configurable dynamics frameskip.
It is not compatible with upstream checkpoint keys. See docs/models/jepa/lewam.md.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_world.losses.sigreg import SIGReg
from torch_world.models.base import WorldActionModel
from torch_world.nn.flow import euler_sample, interpolate, velocity_target
from torch_world.nn.flow import time_embedding as _time_embedding


class _MixtureBlock(nn.Module):
    """Separate state/action projections and MLPs, with shared masked attention."""

    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.norm1, self.norm2 = (
            nn.LayerNorm(dim, elementwise_affine=False),
            nn.LayerNorm(dim, elementwise_affine=False),
        )
        self.qkv = nn.ModuleList([nn.Linear(dim, 3 * dim) for _ in range(2)])
        self.out = nn.ModuleList([nn.Linear(dim, dim) for _ in range(2)])
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_dim, dim))
                for _ in range(2)
            ]
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    @staticmethod
    def _streams(layers, x, split):
        return torch.cat((layers[0](x[:, :split]), layers[1](x[:, split:])), dim=1)

    def forward(self, x, mask, state_count, action_start, condition):
        # Only noisy-action slots receive flow-time/horizon AdaLN conditioning.
        shifts = self.modulation(condition)[:, None].chunk(6, dim=-1)
        sa, ca, ga, sm, cm, gm = [
            torch.cat((torch.full_like(x[:, :action_start], default), value.expand_as(x[:, action_start:])), dim=1)
            for value, default in zip(shifts, (0, 0, 1, 0, 0, 1))
        ]
        h = self.norm1(x) * (1 + ca) + sa
        b, n, d = h.shape
        q, k, v = (
            self._streams(self.qkv, h, state_count).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        )
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0)
        x = x + ga * self._streams(self.out, h.transpose(1, 2).reshape(b, n, d), state_count)
        return x + gm * self._streams(self.mlp, self.norm2(x) * (1 + cm) + sm, state_count)


class LeWAM(WorldActionModel):
    """Joint latent dynamics and flow-matching action chunks (Fu et al., 2026).

    A mixture of transformers predicts future latent states from clean actions,
    and an action velocity field from noisy actions. Its grouped attention mask
    prevents clean-action and future-state leakage into the policy branch.
    Training jointly minimizes dynamics MSE, action flow matching and SIGReg.

    ``encoder`` maps ``(N,C,H,W)`` to ``(N,feature_dim or embed_dim)``. Inputs to ``forward``
    contain ``history + chunk_size`` frames and at least one fewer actions.
    Goals are separate images from the same episode. ``goal_conditioned=False``
    trains task-only behavior cloning. Optional ``remaining`` is a normalized
    remaining horizon, seen only by the action branch.

    ``state_dim`` optionally appends externally normalized measured state. That
    extension predicts state residuals, initialized to zero, and excludes state
    coordinates from SIGReg. ``action_scale`` sets per-coordinate physical action
    units; without statistics, flow matching divides actions by these scales.
    Supply training-only ``action_mean``/``action_std`` to use z-scored actions.
    Sampling clips physical controls after denormalization.

    ``frameskip`` spaces dynamics targets, ``num_states`` sets their count,
    and ``chunk_size`` sets the number of consecutive predicted actions. Use
    ``forward_windows`` when these lengths differ. ``projector_norm="batch"``
    selects the reference BatchNorm projector and positional initialization;
    legacy defaults retain their parameter keys and inference behavior.

    Paper: https://arxiv.org/abs/2610.12818
    Reference: https://github.com/MinghaoFu/lewam
    """

    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int,
        embed_dim: int = 192,
        *,
        history: int = 3,
        chunk_size: int = 8,
        dim: int = 192,
        depth: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 768,
        dropout: float = 0.0,
        flow_steps: int = 8,
        goal_conditioned: bool = True,
        goal_type: str | None = None,
        state_dim: int = 0,
        action_scale=None,
        action_mean=None,
        action_std=None,
        frameskip: int = 1,
        num_states: int | None = None,
        feature_dim: int | None = None,
        projector_norm: str | None = None,
        lamb: float = 0.04,
        action_weight: float = 1.0,
        state_weight: float = 1.0,
        sigreg: SIGReg | None = None,
    ):
        super().__init__()
        if min(action_dim, embed_dim, history, chunk_size, dim, depth, num_heads, mlp_dim, flow_steps) < 1:
            raise ValueError("Model dimensions and flow_steps must be positive")
        if dim % num_heads or dim % 2 or state_dim < 0 or min(lamb, action_weight, state_weight) < 0:
            raise ValueError("Invalid attention dimensions, state_dim or loss weights")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        num_states = chunk_size if num_states is None else num_states
        if frameskip < 1 or num_states < 1 or num_states * frameskip > chunk_size:
            raise ValueError("chunk_size must cover num_states * frameskip")
        if projector_norm not in (None, "batch"):
            raise ValueError("projector_norm must be None or 'batch'")
        goal_type = ("terminal" if goal_conditioned else "none") if goal_type is None else goal_type
        if goal_type not in ("none", "terminal", "sampled") or (goal_type != "none") != goal_conditioned:
            raise ValueError("goal_type must agree with goal_conditioned")
        self.goal_type = goal_type
        self.frameskip, self.num_states = frameskip, num_states
        self.num_clean_actions = num_states * frameskip
        self.encoder = encoder
        self.history, self.chunk_size, self.action_dim = history, chunk_size, action_dim
        self.embed_dim, self.state_dim, self.dim = embed_dim, state_dim, dim
        self.flow_steps, self.goal_conditioned = flow_steps, goal_conditioned
        self.lamb, self.action_weight, self.state_weight = lamb, action_weight, state_weight
        latent_dim = embed_dim + state_dim
        scale = torch.ones(action_dim) if action_scale is None else torch.as_tensor(action_scale, dtype=torch.float32)
        if scale.shape != (action_dim,) or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("action_scale must contain one finite positive scale per action coordinate")
        self.register_buffer("action_scale", scale.clone())
        if (action_mean is None) != (action_std is None):
            raise ValueError("Supply both action_mean and action_std")
        for name, value in (("action_mean", action_mean), ("action_std", action_std)):
            value = None if value is None else torch.as_tensor(value, dtype=torch.float32).clone()
            if value is not None and (value.shape != (action_dim,) or not torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain one finite value per action coordinate")
            if name == "action_std" and value is not None and (value <= 0).any():
                raise ValueError("action_std must be positive")
            self.register_buffer(name, value)
        layers = [nn.Linear(feature_dim or embed_dim, 2 * embed_dim)]
        if projector_norm == "batch":
            layers.append(nn.BatchNorm1d(2 * embed_dim))
        self.projector = nn.Sequential(*layers, nn.GELU(), nn.Linear(2 * embed_dim, embed_dim))
        self.history_projection, self.goal_projection = nn.Linear(latent_dim, dim), nn.Linear(latent_dim, dim)
        self.action_projection = nn.Linear(action_dim, dim)
        self.history_position = nn.Parameter(torch.randn(1, history, dim) * 0.02)
        self.state_queries = nn.Parameter(torch.randn(1, num_states, dim) * 0.02)
        self.action_position = nn.Parameter(torch.randn(1, chunk_size, dim) * 0.02)
        # Reference initialization; legacy defaults retain their checkpoint layout.
        if projector_norm == "batch":
            nn.init.zeros_(self.history_position)
            nn.init.zeros_(self.action_position)
        self.register_parameter(
            "state_position", nn.Parameter(torch.zeros(1, num_states, dim)) if projector_norm == "batch" else None
        )
        self.token_type = nn.Embedding(5, dim)
        self.time_projection, self.horizon_projection = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([_MixtureBlock(dim, num_heads, mlp_dim, dropout) for _ in range(depth)])
        self.state_output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, latent_dim))
        self.action_output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, action_dim))
        if state_dim:
            with torch.no_grad():
                self.state_output[-1].weight[embed_dim:].zero_()
                self.state_output[-1].bias[embed_dim:].zero_()
        self.sigreg = sigreg or SIGReg()

    def encode(self, pixels: Tensor, states: Tensor | None = None) -> Tensor:
        """Encode ``(B,T,C,H,W)`` images, optionally appending ``(B,T,state_dim)``."""
        if pixels.ndim != 5:
            raise ValueError("pixels must have shape (B,T,C,H,W)")
        z = self.projector(self.encoder(pixels.flatten(0, 1))).unflatten(0, pixels.shape[:2])
        if self.state_dim:
            if states is None or states.shape != (*pixels.shape[:2], self.state_dim):
                raise ValueError("Measured states must match batch, time and state_dim")
            z = torch.cat((z, states), dim=-1)
        return z

    def attention_mask(self, history: int, *, device=None) -> Tensor:
        """Boolean SDPA mask: history | future queries | goal | clean | noisy."""
        if not 1 <= history <= self.history:
            raise ValueError("Invalid history length")
        h, s, c, k, g = history, self.num_states, self.num_clean_actions, self.chunk_size, int(self.goal_conditioned)
        clean, noisy = h + s + g, h + s + g + c
        mask = torch.zeros(noisy + k, noisy + k, dtype=torch.bool, device=device)
        mask[:, :h] = True
        mask[h : h + s, h : h + s] = torch.ones(s, s, dtype=torch.bool, device=device).tril()
        for step in range(s):
            mask[h + step, clean : clean + (step + 1) * self.frameskip] = True
        if g:
            mask[h + s, h + s] = True
            mask[noisy:, h + s] = True
        mask[clean:noisy, clean:noisy] = torch.ones(c, c, dtype=torch.bool, device=device).tril()
        mask[noisy:, noisy:] = torch.ones(k, k, dtype=torch.bool, device=device).tril()
        return mask

    def joint_prediction(
        self,
        z: Tensor,
        clean_actions: Tensor,
        noisy_actions: Tensor,
        time: Tensor,
        goal: Tensor | None = None,
        remaining: Tensor | None = None,
        history_pad: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return future latents and action velocities; actions here use normalized units.

        Exposed for leakage tests and advanced training. Clean actions never
        influence action velocities; goal/time/noisy actions never influence dynamics.
        """
        b, h, _ = z.shape
        k = self.chunk_size
        if not 1 <= h <= self.history or z.shape[-1] != self.embed_dim + self.state_dim:
            raise ValueError("Invalid latent history shape")
        if clean_actions.shape != (b, k, self.action_dim) or noisy_actions.shape != clean_actions.shape:
            raise ValueError("Expected full clean and noisy action chunks")
        if time.shape != (b,) or (remaining is not None and remaining.shape != (b,)):
            raise ValueError("time and remaining must have shape (B,)")
        history = self.history_projection(z) + self.history_position[:, -h:] + self.token_type.weight[0]
        future = self.state_queries.expand(b, -1, -1) + self.token_type.weight[1]
        if self.state_position is not None:
            future = future + self.state_position
        tokens = [history, future]
        if self.goal_conditioned:
            # Dynamics-only calls may omit the goal; its token is unreachable
            # from history, clean actions and future queries in every layer.
            goal = torch.zeros_like(z[:, -1]) if goal is None else goal
            if goal.shape != z[:, -1].shape:
                raise ValueError("Goal latent must have shape (B, latent_dim)")
            tokens.append(self.goal_projection(goal)[:, None] + self.token_type.weight[2])
        elif goal is not None:
            raise ValueError("This model is not goal-conditioned")
        clean = (
            self.action_projection(clean_actions[:, : self.num_clean_actions])
            + self.action_position[:, : self.num_clean_actions]
            + self.token_type.weight[3]
        )
        noisy = self.action_projection(noisy_actions) + self.action_position + self.token_type.weight[4]
        state_count = h + self.num_states + int(self.goal_conditioned)
        x = torch.cat((*tokens, clean, noisy), dim=1)
        condition = self.time_projection(_time_embedding(time, self.dim).to(z.dtype))
        if remaining is not None:
            condition = condition + self.horizon_projection(_time_embedding(remaining, self.dim).to(z.dtype))
        mask = self.attention_mask(h, device=z.device)
        if history_pad is not None:
            if history_pad.shape != (b, h) or history_pad.dtype != torch.bool or history_pad[:, -1].any():
                raise ValueError("history_pad must be boolean (B,H), with the current frame unpadded")
            mask = mask[None, None].expand(b, 1, -1, -1).clone()
            mask[:, :, :, :h] &= ~history_pad[:, None, None, :]
        for block in self.blocks:
            x = block(x, mask, state_count, state_count + self.num_clean_actions, condition)
        predicted = self.state_output(x[:, h : h + self.num_states])
        if self.state_dim:
            predicted = torch.cat(
                (predicted[..., : self.embed_dim], z[:, -1:, self.embed_dim :] + predicted[..., self.embed_dim :]),
                dim=-1,
            )
        return predicted, self.action_output(x[:, -k:])

    def normalize_actions(self, actions: Tensor) -> Tensor:
        """Physical controls to training coordinates (z-score when statistics exist)."""
        if self.action_mean is not None:
            return (actions - self.action_mean) / self.action_std
        return actions / self.action_scale

    def denormalize_actions(self, actions: Tensor, *, clamp: bool = False) -> Tensor:
        """Training coordinates to physical controls; bounds apply after conversion."""
        raw = actions * self.action_scale if self.action_mean is None else actions * self.action_std + self.action_mean
        return raw.maximum(-self.action_scale).minimum(self.action_scale) if clamp else raw

    def predict_future(
        self, z: Tensor, actions: Tensor, goal: Tensor | None = None, *, history_pad: Tensor | None = None
    ) -> Tensor:
        """Predict a chunk of future latents from physical-unit actions, differentiably.

        ``goal`` is accepted for the ``WorldActionModel`` interface and ignored: LeWAM's dynamics never see the goal.
        """
        if actions.shape != (z.size(0), self.chunk_size, self.action_dim):
            raise ValueError("predict_future requires a complete action chunk")
        return self.joint_prediction(
            z,
            self.normalize_actions(actions),
            torch.zeros_like(actions),
            z.new_zeros(z.size(0)),
            history_pad=history_pad,
        )[0]

    def predict(self, z: Tensor, actions: Tensor) -> Tensor:
        """WorldModel-compatible next-state predictions for each history prefix."""
        if self.frameskip != 1:
            raise ValueError("predict is a one-step API; use predict_future for frameskip dynamics")
        if actions.shape != (*z.shape[:2], self.action_dim):
            raise ValueError("Actions must align with latent history")
        predictions = []
        for t in range(z.size(1)):
            chunk = F.pad(actions[:, t : t + 1], (0, 0, 0, self.chunk_size - 1))
            predictions.append(self.predict_future(z[:, max(0, t + 1 - self.history) : t + 1], chunk)[:, :1])
        return torch.cat(predictions, dim=1)

    def sample_actions(
        self,
        z: Tensor,
        goal: Tensor | None = None,
        *,
        remaining: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
        steps: int | None = None,
        clamp: bool = True,
        history_pad: Tensor | None = None,
    ) -> Tensor:
        """Euler-integrate noise → action flow; return bounded physical-unit chunks."""
        if self.goal_conditioned and goal is None:
            raise ValueError("Goal-conditioned action sampling requires a goal latent")
        steps = self.flow_steps if steps is None else steps
        if steps < 1:
            raise ValueError("steps must be positive")
        shape = (z.size(0), self.chunk_size, self.action_dim)
        actions = torch.randn(shape, device=z.device, dtype=z.dtype, generator=generator) if noise is None else noise
        if actions.shape != shape:
            raise ValueError("Noise must have shape (B, chunk_size, action_dim)")
        clean = torch.zeros_like(actions)
        actions = euler_sample(
            lambda x, time: self.joint_prediction(z, clean, x, time, goal, remaining, history_pad)[1], actions, steps
        )
        return self.denormalize_actions(actions, clamp=clamp)

    def _distance(self, predicted, target):
        error = (predicted - target).square()
        distance = error[..., : self.embed_dim].mean(-1)
        if self.state_dim:
            distance = distance + self.state_weight * error[..., self.embed_dim :].mean(-1)
        return distance

    def cost(self, z: Tensor, goal: Tensor) -> Tensor:
        return self._distance(z[:, -1], goal)

    def forward(
        self,
        pixels: Tensor,
        actions: Tensor,
        states: Tensor | None = None,
        *,
        goal_pixels: Tensor | None = None,
        goal_states: Tensor | None = None,
        action_valid: Tensor | None = None,
        remaining: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        """Train on aligned windows; action_valid masks failed demonstrations (B,K).

        Targets stay attached to the jointly trained encoder. All valid windows
        train dynamics; only action_valid entries supervise the action branch.
        An optional final padded action is ignored. Goals must be supplied
        explicitly, avoiding an implicit short-horizon/final-goal mismatch.
        """
        if self.frameskip != 1 or self.num_states != self.chunk_size:
            raise ValueError("Use forward_windows for spaced dynamics targets")
        h, k = self.history, self.chunk_size
        if pixels.size(1) != h + k or actions.shape[:2] not in ((pixels.size(0), h + k - 1), (pixels.size(0), h + k)):
            raise ValueError("Expected history + chunk_size frames and aligned actions")
        z = self.encode(pixels, states)
        goal = None
        if self.goal_conditioned:
            if goal_pixels is None:
                raise ValueError("Training requires explicit goal_pixels")
            goal = self.encode(goal_pixels[:, None], None if goal_states is None else goal_states[:, None])[:, 0]
        target = self.normalize_actions(actions[:, h - 1 : h + k - 1])
        noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
        time = torch.rand(z.size(0), device=z.device, dtype=z.dtype, generator=generator)
        noisy = interpolate(noise, target, time)
        predicted, velocity = self.joint_prediction(z[:, :h], target, noisy, time, goal, remaining)
        dynamics = self._distance(predicted, z[:, h:]).mean()
        error = (velocity - velocity_target(noise, target)).square().mean(-1)
        if action_valid is None:
            action_valid = torch.ones_like(error)
        if action_valid.shape != error.shape or not torch.isfinite(action_valid).all() or (action_valid < 0).any():
            raise ValueError("action_valid must be finite nonnegative weights with shape (B,K)")
        action_loss = ((error * action_valid).sum(-1) / action_valid.sum(-1).clamp_min(1)).mean()
        visual = z[..., : self.embed_dim]
        if goal is not None:
            visual = torch.cat((visual, goal[:, None, : self.embed_dim]), dim=1)
        regularizer = self.sigreg(visual.transpose(0, 1), generator=generator)
        return {
            "loss": dynamics + self.action_weight * action_loss + self.lamb * regularizer,
            "pred_loss": dynamics,
            "action_loss": action_loss,
            "sigreg_loss": regularizer,
        }

    def forward_windows(
        self,
        history_pixels: Tensor,
        future_pixels: Tensor,
        actions: Tensor,
        *,
        goal_pixels: Tensor | None = None,
        history_pad: Tensor | None = None,
        action_valid: Tensor | None = None,
        state_valid: Tensor | None = None,
        remaining: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        """Reference-aligned single-view MSE-state/flow-action training.

        History contains H strided frames, future contains S frames at offsets
        K,2K,...,S*K, and actions contains C consecutive physical controls from
        the current observation. True history_pad keys are excluded. Invalid
        actions are zero in normalized coordinates. Losses average valid tokens
        within each example before averaging examples, as in the reference.
        All images share one encoder/projector call; targets are not detached.
        """
        if self.state_dim:
            raise ValueError("forward_windows is the visual-only reference path")
        b = history_pixels.size(0)
        if history_pixels.shape[:2] != (b, self.history) or future_pixels.shape[:2] != (b, self.num_states):
            raise ValueError("Expected H history frames and S future frames")
        if actions.shape != (b, self.chunk_size, self.action_dim):
            raise ValueError("Expected C consecutive physical actions")
        images = [history_pixels, future_pixels]
        if self.goal_conditioned:
            if goal_pixels is None:
                raise ValueError("Training requires explicit goal_pixels")
            images.append(goal_pixels[:, None])
        elif goal_pixels is not None:
            raise ValueError("This model is not goal-conditioned")
        z = self.encode(torch.cat(images, 1))
        goal = z[:, -1] if self.goal_conditioned else None
        target = self.normalize_actions(actions)
        if action_valid is not None:
            target = target.masked_fill(~action_valid.bool()[..., None], 0)
        noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
        time = torch.rand(b, device=z.device, dtype=z.dtype, generator=generator)
        predicted, velocity = self.joint_prediction(
            z[:, : self.history],
            target,
            interpolate(noise, target, time),
            time,
            goal,
            remaining,
            history_pad,
        )

        def masked_mean(error, valid):
            if valid is None:
                return error.mean()
            if valid.shape != error.shape or not torch.isfinite(valid).all() or (valid < 0).any():
                raise ValueError("Validity weights must be finite, nonnegative and match token errors")
            return ((error * valid).sum(-1) / valid.sum(-1).clamp_min(1)).mean()

        dynamics = masked_mean(
            self._distance(predicted, z[:, self.history : self.history + self.num_states]), state_valid
        )
        action_loss = masked_mean((velocity - velocity_target(noise, target)).square().mean(-1), action_valid)
        # Independent projection draws for each time/goal group, matching upstream.
        regularizer = torch.stack([self.sigreg(group[None], generator=generator) for group in z.unbind(1)]).mean()
        return {
            "loss": dynamics + self.action_weight * action_loss + self.lamb * regularizer,
            "pred_loss": dynamics,
            "action_loss": action_loss,
            "sigreg_loss": regularizer,
        }
