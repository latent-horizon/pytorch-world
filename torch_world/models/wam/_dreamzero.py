import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_world.models.base import WorldActionModel
from torch_world.nn.attention import flex_attention
from torch_world.nn.flow import euler_sample, interpolate, sample_time, time_embedding, velocity_target
from torch_world.nn.positional import sincos_1d
from torch_world.nn.transformer import MLP, AdaLNBlock, init_weights

_ANCHOR, _GOAL, _VIDEO, _ACTION, _STATE = range(5)


def _sincos_3d(dim: int, coords: Tensor) -> Tensor:
    """``sincos_3d`` at arbitrary ``(N, 3)`` coordinates."""
    sub = 2 * math.ceil(dim / 6)
    return torch.cat([sincos_1d(sub, coords[:, i]) for i in range(3)], dim=-1)[:, :dim]


class DreamZero(WorldActionModel):
    """Autoregressive joint video-action diffusion transformer (Ye et al., 2026).

    A World Action Model: one causal DiT denoises the next chunk of video latents *and* the action chunk that
    produces them with a single flow-matching objective. Chunk ``k`` holds ``chunk_frames`` latent frames, the
    ``chunk_actions`` actions executed over them and, with ``state_dim``, the proprioceptive state ``q_k`` at its
    start. Training uses chunk-wise teacher forcing: every chunk of a clip is noised with its own flow time and attends
    to an anchor frame (and a goal frame when ``goal_conditioned``), the *clean* earlier chunks and itself. Video and
    actions of a chunk share their time, except with ``flash_alpha``, which draws video times from
    ``1 - Beta(flash_alpha, 1)`` (DreamZero-Flash) so that few-step sampling denoises actions from still-noisy video.
    The loss is ``video_weight * flow_video + action_weight * flow_action``; ``video_weight=0`` keeps the architecture
    but trains a pure policy, the ablation that measures what predicting video adds. ``time_schedule="logit_normal"``
    draws shared times from ``sigmoid(N(0, 1))`` instead of the paper's ``U(0, 1)``.

    Inputs are latents of a video autoencoder (``ConvVideoAE``, or the Wan ``VideoVAE`` of the paper). ``forward``
    takes ``(B, 1 + num_chunks * chunk_frames, C, h, w)`` latents, ``(B, num_chunks * chunk_actions, A)`` actions in
    environment units and optional ``(B, num_chunks, state_dim)`` states. At inference ``imagine`` samples the next
    chunk from the latest ``num_chunks - 1`` past chunks; ``DreamZeroPolicy`` runs it in closed loop, replacing the
    predicted latents in that context with encoded real observations after each chunk.

    ``action_scale`` sets per-coordinate physical action units; ``action_mean``/``action_std`` z-score actions instead.
    Sampled actions are clipped to ``±action_scale``.

    ``attention="flex"`` runs the block-causal mask as a FlexAttention ``BlockMask``, compiled on CUDA, so masked-out
    blocks are skipped instead of materialized; masks are cached per sequence layout. Sampling computes the keys and
    values of the clean context once per chunk (``context_cache``) and denoises only the new chunk at every step.

    Paper: https://arxiv.org/abs/2602.15922
    """

    history = None

    def __init__(
        self,
        latent_channels: int,
        latent_size: tuple[int, int],
        action_dim: int,
        *,
        state_dim: int = 0,
        patch_size: int = 1,
        chunk_frames: int = 2,
        chunk_actions: int = 8,
        num_chunks: int = 4,
        dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_dim: int | None = None,
        dropout: float = 0.0,
        flow_steps: int = 16,
        goal_conditioned: bool = False,
        action_weight: float = 1.0,
        video_weight: float = 1.0,
        time_schedule: str = "uniform",
        flash_alpha: float | None = None,
        action_scale=None,
        action_mean=None,
        action_std=None,
        autoencoder: nn.Module | None = None,
        attention: str = "sdpa",
    ):
        super().__init__()
        h, w = latent_size
        if h % patch_size or w % patch_size:
            raise ValueError("latent_size must be divisible by patch_size")
        if min(latent_channels, action_dim, chunk_frames, chunk_actions, num_chunks, flow_steps) < 1 or state_dim < 0:
            raise ValueError("Model sizes and flow_steps must be positive")
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        if flash_alpha is not None and flash_alpha <= 0:
            raise ValueError("flash_alpha must be positive")
        if time_schedule not in ("uniform", "logit_normal") or min(action_weight, video_weight) < 0:
            raise ValueError("time_schedule must be 'uniform' or 'logit_normal', with nonnegative loss weights")
        if attention not in ("sdpa", "flex"):
            raise ValueError("attention must be 'sdpa' or 'flex'")
        self.attention, self._masks = attention, {}
        self.latent_channels, self.latent_size, self.patch_size = latent_channels, (h, w), patch_size
        self.grid = (h // patch_size, w // patch_size)
        self.tokens_per_frame = self.grid[0] * self.grid[1]
        self.action_dim, self.state_dim = action_dim, state_dim
        self.chunk_frames, self.chunk_size, self.num_chunks = chunk_frames, chunk_actions, num_chunks
        self.dim, self.flow_steps, self.goal_conditioned = dim, flow_steps, goal_conditioned
        self.action_weight, self.video_weight, self.flash_alpha = action_weight, video_weight, flash_alpha
        self.time_schedule = time_schedule
        self.autoencoder = autoencoder
        scale = torch.ones(action_dim) if action_scale is None else torch.as_tensor(action_scale, dtype=torch.float32)
        if scale.shape != (action_dim,) or (scale <= 0).any():
            raise ValueError("action_scale must hold one positive scale per action coordinate")
        if (action_mean is None) != (action_std is None):
            raise ValueError("Supply both action_mean and action_std")
        self.register_buffer("action_scale", scale.clone())
        self.register_buffer("action_mean", None if action_mean is None else torch.as_tensor(action_mean).float())
        self.register_buffer("action_std", None if action_std is None else torch.as_tensor(action_std).float())

        patch_dim = latent_channels * patch_size**2
        self.video_in, self.video_out = nn.Linear(patch_dim, dim), nn.Linear(dim, patch_dim)
        self.action_encoder = MLP([action_dim, dim, dim], act=nn.SiLU)
        self.action_decoder = MLP([dim, dim, action_dim], act=nn.SiLU)
        self.state_encoder = MLP([state_dim, dim, dim], act=nn.SiLU) if state_dim else None
        self.token_type = nn.Embedding(5, dim)
        self.time_mlp = MLP([dim, dim, dim], act=nn.SiLU)
        self.blocks = nn.ModuleList(
            AdaLNBlock(dim, num_heads, mlp_dim=mlp_dim, dropout=dropout, act=nn.SiLU) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        init_weights(self.video_in)
        init_weights(self.blocks)
        for module in [*(b.modulation[-1] for b in self.blocks), self.final_modulation[-1]]:
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    # Actions and latents ----------------------------------------------------------------------------------------

    def normalize_actions(self, actions: Tensor) -> Tensor:
        """Physical controls to training coordinates (z-score when statistics exist)."""
        if self.action_mean is not None:
            return (actions - self.action_mean) / self.action_std
        return actions / self.action_scale

    def denormalize_actions(self, actions: Tensor, *, clamp: bool = False) -> Tensor:
        """Training coordinates to physical controls; bounds apply after conversion."""
        raw = actions * self.action_scale if self.action_mean is None else actions * self.action_std + self.action_mean
        return raw.maximum(-self.action_scale).minimum(self.action_scale) if clamp else raw

    def encode(self, pixels: Tensor) -> Tensor:
        """Video ``(B, T, 3, H, W)`` in ``[0, 1]`` to latents ``(B, T', C, h, w)`` with the autoencoder."""
        if self.autoencoder is None:
            raise ValueError("encode needs an autoencoder; pass precomputed latents instead")
        return self.autoencoder.encode(pixels)

    def decode(self, z: Tensor) -> Tensor | None:
        return None if self.autoencoder is None else self.autoencoder.decode(z)

    @property
    def raw_frames_per_chunk(self) -> int:
        """Video frames each chunk covers before the autoencoder's temporal compression."""
        return self.chunk_frames * getattr(self.autoencoder, "temporal_ratio", 1)

    # Sequence --------------------------------------------------------------------------------------------------

    def _video_tokens(self, latents: Tensor, first: int, kind: int) -> Tensor:
        """Tokens of ``(B, T, C, h, w)`` latent frames at time indices ``first, first + 1, ...``."""
        B, T = latents.shape[:2]
        x = F.pixel_unshuffle(latents.flatten(0, 1), self.patch_size).flatten(2).transpose(1, 2)
        x = self.video_in(x.unflatten(0, (B, T)).flatten(1, 2))
        axes = (torch.arange(first, first + T), torch.arange(self.grid[0]), torch.arange(self.grid[1]))
        coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).flatten(0, -2)
        return x + _sincos_3d(self.dim, coords).to(x) + self.token_type.weight[kind]

    def _video_output(self, tokens: Tensor) -> Tensor:
        """Tokens ``(B, T * P, D)`` back to latents ``(B, T, C, h, w)``."""
        B = tokens.size(0)
        x = self.video_out(tokens).unflatten(1, (-1, self.tokens_per_frame))
        x = x.flatten(0, 1).transpose(1, 2).unflatten(-1, self.grid)
        return F.pixel_shuffle(x, self.patch_size).unflatten(0, (B, -1))

    def _chunk_tokens(self, video: Tensor, actions: Tensor, state: Tensor | None, chunk: int) -> Tensor:
        index = torch.arange(self.chunk_size) + (chunk - 1) * self.chunk_size
        parts = [
            self._video_tokens(video, 1 + (chunk - 1) * self.chunk_frames, _VIDEO),
            self.action_encoder(actions) + sincos_1d(self.dim, index).to(actions) + self.token_type.weight[_ACTION],
        ]
        if self.state_encoder is not None:
            position = sincos_1d(self.dim, torch.tensor([float(chunk)])).to(actions)
            parts.append(self.state_encoder(state[:, None]) + position + self.token_type.weight[_STATE])
        return torch.cat(parts, dim=1)

    @property
    def _chunk_length(self) -> int:
        return self.chunk_frames * self.tokens_per_frame + self.chunk_size + int(self.state_encoder is not None)

    def _prefix_tokens(self, anchor: Tensor, goal: Tensor | None) -> Tensor:
        if self.goal_conditioned != (goal is not None):
            raise ValueError("Pass a goal latent exactly when the model is goal-conditioned")
        tokens = [self._video_tokens(anchor[:, None], 0, _ANCHOR)]
        if goal is not None:
            tokens.append(self._video_tokens(goal[:, None], 0, _GOAL))
        return torch.cat(tokens, dim=1)

    def _chunks(self, video: Tensor, actions: Tensor, states: Tensor | None, first: int) -> list[Tensor]:
        """Tokens of consecutive chunks ``first, first + 1, ...``; ``states`` is indexed by chunk number."""
        K, H = self.chunk_frames, self.chunk_size
        return [
            self._chunk_tokens(
                video[:, i * K : (i + 1) * K],
                actions[:, i * H : (i + 1) * H],
                None if self.state_encoder is None else states[:, first + i - 1],
                first + i,
            )
            for i in range(video.size(1) // K)
        ]

    def attention_mask(self, clean: int, noisy: int, start: int, device=None):
        """Mask of ``prefix | clean chunks 1..clean | noisy chunks start..``, cached per layout.

        Noisy chunk ``k`` attends to the prefix, clean chunks before ``k`` and itself; clean chunk ``k`` to the prefix
        and clean chunks up to ``k``. A boolean ``(N, N)`` tensor, or a FlexAttention ``BlockMask`` with
        ``attention="flex"``.
        """
        key = (clean, noisy, start, str(device))
        if key not in self._masks:
            prefix, L = self.tokens_per_frame * (1 + int(self.goal_conditioned)), self._chunk_length
            chunk = [0] * prefix + [k for k in range(1, clean + 1) for _ in range(L)]
            chunk += [k for k in range(start, start + noisy) for _ in range(L)]
            chunk = torch.tensor(chunk, device=device)
            is_noisy = torch.arange(len(chunk), device=device) >= prefix + clean * L

            def allowed(q, kv):
                clean_key = ~is_noisy[kv] & (chunk[kv] <= chunk[q] - is_noisy[q].long())
                return clean_key | (is_noisy[kv] & is_noisy[q] & (chunk[kv] == chunk[q]))

            if self.attention == "flex":
                from torch.nn.attention.flex_attention import create_block_mask

                n = len(chunk)
                mask = create_block_mask(lambda b, h, q, kv: allowed(q, kv), None, None, n, n, device=device)
            else:
                index = torch.arange(len(chunk), device=device)
                mask = allowed(index[:, None], index[None])
            self._masks[key] = mask
        return self._masks[key]

    def _condition(self, time: Tensor, like: Tensor) -> Tensor:
        return self.time_mlp(time_embedding(time, self.dim).to(like))

    def _noisy_time(self, t_video: Tensor, t_action: Tensor) -> Tensor:
        """Per-token times ``(B, n * L)`` of noisy chunks: video tokens at ``t_video``, the rest at ``t_action``."""
        video = self.chunk_frames * self.tokens_per_frame
        return torch.cat(
            (t_video[..., None].expand(-1, -1, video), t_action[..., None].expand(-1, -1, self._chunk_length - video)),
            dim=-1,
        ).flatten(1)

    def _output(self, x: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        """Velocities from the final tokens ``(B, n * L, D)`` of ``n`` noisy chunks."""
        K, H, P = self.chunk_frames, self.chunk_size, self.tokens_per_frame
        shift, scale = self.final_modulation(c).chunk(2, dim=-1)
        x = (self.norm(x) * (1 + scale) + shift).unflatten(1, (-1, self._chunk_length))
        return self._video_output(x[:, :, : K * P].flatten(1, 2)), self.action_decoder(
            x[:, :, K * P : K * P + H]
        ).flatten(1, 2)

    def _check(self, anchor, video, actions, noisy_video, noisy_actions, t_video, t_action, start, states):
        K, H = self.chunk_frames, self.chunk_size
        j, n = video.size(1) // K, noisy_video.size(1) // K
        if video.size(1) != j * K or actions.size(1) != j * H or noisy_video.size(1) != n * K or n < 1:
            raise ValueError("Video and actions must hold whole chunks")
        if noisy_actions.size(1) != n * H or t_video.shape != (anchor.size(0), n) or t_action.shape != t_video.shape:
            raise ValueError("Noisy actions and times must match the noisy chunks")
        if start not in (1, j + 1):
            raise ValueError("Noisy chunks start at chunk 1 (teacher forcing) or right after the clean ones")
        if self.state_encoder is not None and (states is None or states.size(1) < max(j, start + n - 1)):
            raise ValueError("Pass one state per chunk")
        return j, n

    def joint_velocity(
        self,
        anchor: Tensor,
        video: Tensor,
        actions: Tensor,
        noisy_video: Tensor,
        noisy_actions: Tensor,
        t_video: Tensor,
        t_action: Tensor,
        *,
        start: int,
        states: Tensor | None = None,
        goal: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Video and action velocities of ``n`` noisy chunks numbered ``start, ..., start + n - 1``.

        The sequence is ``anchor (| goal) | clean chunks 1..j | noisy chunks``, with ``anchor`` ``(B, C, h, w)`` the
        latent frame before chunk 1, clean ``video`` ``(B, j * K, C, h, w)`` and normalized ``actions``
        ``(B, j * H, A)``. Noisy chunks are ``(B, n * K, C, h, w)`` and ``(B, n * H, A)`` with times ``(B, n)``.
        ``states`` ``(B, max(j, start + n - 1), S)`` holds ``q_k`` for every chunk ``k``; see ``attention_mask``.
        Training passes ``j = n - 1`` and ``start = 1``; sampling the next chunk passes ``n = 1`` and ``start = j + 1``.
        """
        j, n = self._check(anchor, video, actions, noisy_video, noisy_actions, t_video, t_action, start, states)
        prefix = self._prefix_tokens(anchor, goal)
        x = torch.cat(
            [
                prefix,
                *self._chunks(video, actions, states, 1),
                *self._chunks(noisy_video, noisy_actions, states, start),
            ],
            dim=1,
        )
        context = prefix.size(1) + j * self._chunk_length
        time = torch.cat((x.new_ones(x.size(0), context), self._noisy_time(t_video, t_action).to(x)), dim=1)
        c = self._condition(time, x)
        mask = self.attention_mask(j, n, start, x.device)
        for block in self.blocks:
            x = block(x, c, attn_mask=mask)
        return self._output(x[:, context:], c[:, context:])

    # Key-value cache -------------------------------------------------------------------------------------------

    @staticmethod
    def _block(block: AdaLNBlock, x: Tensor, c: Tensor, mask=None, cache: tuple[Tensor, Tensor] | None = None):
        """``AdaLNBlock.forward`` (in eval) that prepends cached keys and values and returns its own."""
        shift1, scale1, gate1, shift2, scale2, gate2 = block.modulation(c).chunk(6, dim=-1)
        attn, (B, N, _) = block.attn, x.shape
        q, k, v = (
            attn.qkv(block.norm1(x) * (1 + scale1) + shift1).view(B, N, 3, attn.num_heads, -1).permute(2, 0, 3, 1, 4)
        )
        keys, values = (k, v) if cache is None else (torch.cat((cache[0], k), 2), torch.cat((cache[1], v), 2))
        if mask is not None and not isinstance(mask, Tensor):
            out = flex_attention(q, keys, values, mask)
        else:
            out = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask)
        x = x + gate1 * attn.proj(out.transpose(1, 2).reshape(B, N, -1))
        return x + gate2 * block.mlp(block.norm2(x) * (1 + scale2) + shift2), (k, v)

    def context_cache(
        self,
        anchor: Tensor,
        video: Tensor,
        actions: Tensor,
        *,
        states: Tensor | None = None,
        goal: Tensor | None = None,
    ) -> list[tuple[Tensor, Tensor]]:
        """Per-layer keys and values of ``anchor (| goal) | clean chunks``, computed once per chunk.

        Clean tokens never attend to noisy ones, so the context is the same at every denoising step of the next chunk.
        """
        prefix = self._prefix_tokens(anchor, goal)
        x = torch.cat([prefix, *self._chunks(video, actions, states, 1)], dim=1)
        c = self._condition(x.new_ones(x.shape[:2]), x)
        mask = self.attention_mask(video.size(1) // self.chunk_frames, 0, 1, x.device)
        cache = []
        for block in self.blocks:
            x, kv = self._block(block, x, c, mask)
            cache.append(kv)
        return cache

    def cached_velocity(
        self,
        cache: list[tuple[Tensor, Tensor]],
        noisy_video: Tensor,
        noisy_actions: Tensor,
        t_video: Tensor,
        t_action: Tensor,
        *,
        start: int,
        states: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """``joint_velocity`` of the one chunk after the cached context; it attends to all of it."""
        x = torch.cat(self._chunks(noisy_video, noisy_actions, states, start), dim=1)
        c = self._condition(self._noisy_time(t_video, t_action).to(x), x)
        for block, kv in zip(self.blocks, cache):
            x = self._block(block, x, c, cache=kv)[0]
        return self._output(x, c)

    # Training --------------------------------------------------------------------------------------------------

    def forward(
        self,
        latents: Tensor,
        actions: Tensor,
        states: Tensor | None = None,
        *,
        goal: Tensor | None = None,
        action_valid: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        """Teacher-forced joint flow matching over ``num_chunks`` chunks.

        ``latents`` ``(B, 1 + M * K, C, h, w)``: an anchor frame then ``M`` chunks of ``K`` frames. ``actions``
        ``(B, M * H, A)`` in environment units, ``states`` ``(B, M, S)``, ``goal`` latent ``(B, C, h, w)``.
        ``action_valid`` ``(B, M * H)`` weights the action loss, for example to drop padded actions.
        """
        B, M, K, H = latents.size(0), self.num_chunks, self.chunk_frames, self.chunk_size
        if latents.size(1) != 1 + M * K or actions.shape != (B, M * H, self.action_dim):
            raise ValueError("Expected 1 + num_chunks * chunk_frames latents and num_chunks * chunk_actions actions")
        video, target = latents[:, 1:], self.normalize_actions(actions)
        schedule = (0.0, 1.0) if self.time_schedule == "logit_normal" else None
        t_action = sample_time(
            (B, M), device=latents.device, dtype=latents.dtype, generator=generator, logit_normal=schedule
        )
        t_video = (
            t_action
            if self.flash_alpha is None
            else sample_time(
                (B, M), device=latents.device, dtype=latents.dtype, generator=generator, alpha=self.flash_alpha
            )
        )
        video_noise = torch.randn(video.shape, device=video.device, dtype=video.dtype, generator=generator)
        action_noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
        noisy_video = interpolate(video_noise, video, t_video.repeat_interleave(K, 1))
        noisy_actions = interpolate(action_noise, target, t_action.repeat_interleave(H, 1))
        v_video, v_action = self.joint_velocity(
            latents[:, 0],
            video[:, : (M - 1) * K],
            target[:, : (M - 1) * H],
            noisy_video,
            noisy_actions,
            t_video,
            t_action,
            start=1,
            states=states,
            goal=goal,
        )
        video_loss = (v_video - velocity_target(video_noise, video)).square().mean()
        error = (v_action - velocity_target(action_noise, target)).square().mean(-1)
        if action_valid is None:
            action_loss = error.mean()
        else:
            if action_valid.shape != error.shape:
                raise ValueError("action_valid must have shape (B, num_chunks * chunk_actions)")
            action_loss = (error * action_valid).sum() / action_valid.sum().clamp_min(1)
        return {
            "loss": self.video_weight * video_loss + self.action_weight * action_loss,
            "video_loss": video_loss,
            "action_loss": action_loss,
        }

    # Inference -------------------------------------------------------------------------------------------------

    def _context(self, z: Tensor, past_actions: Tensor | None, states: Tensor | None):
        """Keep the latest ``num_chunks - 1`` whole chunks; the frame before them becomes the anchor."""
        K, H = self.chunk_frames, self.chunk_size
        j = (z.size(1) - 1) // K
        if z.size(1) != 1 + j * K:
            raise ValueError("Context latents must be an anchor frame then whole chunks")
        past_actions = z.new_zeros(z.size(0), 0, self.action_dim) if past_actions is None else past_actions
        if past_actions.shape != (z.size(0), j * self.chunk_size, self.action_dim):
            raise ValueError("past_actions must hold chunk_actions actions per context chunk")
        if self.state_encoder is not None and (states is None or states.size(1) != j + 1):
            raise ValueError("Pass one state per context chunk plus the current state")
        drop = max(0, j - (self.num_chunks - 1))
        z, past_actions = z[:, drop * K :], past_actions[:, drop * H :]
        states = None if states is None else states[:, drop:]
        return z[:, 0], z[:, 1:], self.normalize_actions(past_actions), states, j - drop

    def _sample(self, z, goal, past_actions, states, steps, generator, noise, video_noise, actions=None, video=None):
        """Integrate the flow of the next chunk's video and actions, keeping whichever is given clean (time 1)."""
        anchor, context, past, states, j = self._context(z, past_actions, states)
        B, K, H = z.size(0), self.chunk_frames, self.chunk_size
        steps = self.flow_steps if steps is None else steps
        shape_v, shape_a = (B, K, self.latent_channels, *self.latent_size), (B, H, self.action_dim)
        if video is None and video_noise is None:
            video_noise = torch.randn(shape_v, device=z.device, dtype=z.dtype, generator=generator)
        if actions is None and noise is None:
            noise = torch.randn(shape_a, device=z.device, dtype=z.dtype, generator=generator)
        if (video is None and video_noise.shape != shape_v) or (actions is None and noise.shape != shape_a):
            raise ValueError("Noise must match one chunk")
        if (video is not None and video.shape != shape_v) or (actions is not None and actions.shape != shape_a):
            raise ValueError("Clean video and actions must be one chunk")
        cache = self.context_cache(anchor, context, past, states=states, goal=goal)
        split = 0 if video is not None else video_noise[0].numel()

        def unpack(x):
            v = video if video is not None else x[:, :split].view(shape_v)
            a = actions if actions is not None else x[:, split:].view(shape_a)
            return v, a

        def velocity(x, time):
            v, a = unpack(x)
            one = time.new_ones(B, 1)
            t_video, t_action = (
                (one if video is not None else time[:, None]),
                (one if actions is not None else time[:, None]),
            )
            dv, da = self.cached_velocity(cache, v, a, t_video, t_action, start=j + 1, states=states)
            return torch.cat([d.flatten(1) for d, given in ((dv, video), (da, actions)) if given is None], dim=1)

        x = torch.cat([n.flatten(1) for n, given in ((video_noise, video), (noise, actions)) if given is None], dim=1)
        return unpack(euler_sample(velocity, x, steps))

    def imagine(
        self,
        z: Tensor,
        goal: Tensor | None = None,
        *,
        past_actions: Tensor | None = None,
        states: Tensor | None = None,
        steps: int | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
        video_noise: Tensor | None = None,
        clamp: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Jointly sample the next chunk: latents ``(B, K, C, h, w)`` and physical actions ``(B, H, A)``.

        ``z`` ``(B, 1 + j * K, C, h, w)`` is an anchor frame then ``j`` past chunks, ``past_actions``
        ``(B, j * H, A)`` the actions executed over them and ``states`` ``(B, j + 1, S)`` the state at the start of
        each past chunk and now. ``noise`` and ``video_noise`` fix the starting Gaussian draws.
        """
        future, actions = self._sample(z, goal, past_actions, states, steps, generator, noise, video_noise)
        return future, self.denormalize_actions(actions, clamp=clamp)

    def inverse_dynamics(
        self,
        z: Tensor,
        future: Tensor,
        goal: Tensor | None = None,
        *,
        past_actions: Tensor | None = None,
        states: Tensor | None = None,
        steps: int | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
        clamp: bool = True,
    ) -> Tensor:
        """Actions ``(B, H, A)`` denoised with the next chunk's video given clean, ``future`` ``(B, K, C, h, w)``.

        A diagnostic: with the true future, its error against the recorded actions bounds what the joint policy can
        reach when its imagined video is perfect.
        """
        normalized = self._sample(z, goal, past_actions, states, steps, generator, noise, None, video=future)[1]
        return self.denormalize_actions(normalized, clamp=clamp)

    def sample_actions(
        self,
        z: Tensor,
        goal: Tensor | None = None,
        *,
        history_pad: Tensor | None = None,
        remaining: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """The policy: the action half of ``imagine``. Takes the same arguments."""
        if history_pad is not None or remaining is not None:
            raise ValueError("DreamZero does not use history_pad or remaining")
        return self.imagine(z, goal, **kwargs)[1]

    def predict_future(
        self,
        z: Tensor,
        actions: Tensor,
        goal: Tensor | None = None,
        *,
        past_actions: Tensor | None = None,
        states: Tensor | None = None,
        steps: int | None = None,
        generator: torch.Generator | None = None,
        video_noise: Tensor | None = None,
        history_pad: Tensor | None = None,
    ) -> Tensor:
        """Forward dynamics: denoise the next chunk's latents given its clean physical ``actions`` ``(B, H, A)``.

        Differentiable in ``actions``, for gradient-based refinement. A goal-conditioned model needs its ``goal``.
        """
        if history_pad is not None:
            raise ValueError("DreamZero does not use history_pad")
        if actions.shape != (z.size(0), self.chunk_size, self.action_dim):
            raise ValueError("predict_future needs one action chunk")
        normalized = self.normalize_actions(actions)
        return self._sample(z, goal, past_actions, states, steps, generator, None, video_noise, normalized)[0]

    def rollout(self, z: Tensor, actions: Tensor, *, goal: Tensor | None = None, **kwargs) -> Tensor:
        """Unroll ``(B, T, A)`` actions, ``T`` a multiple of ``chunk_actions``, chunk by chunk from context ``z``.

        Returns ``z`` followed by the predicted latents of every chunk, so ``cost`` scores the last predicted frame.
        """
        H = self.chunk_size
        if actions.size(1) % H:
            raise ValueError("rollout needs whole action chunks")
        past = kwargs.pop("past_actions", None)
        past = z.new_zeros(z.size(0), 0, self.action_dim) if past is None else past
        for c in range(actions.size(1) // H):
            chunk = actions[:, c * H : (c + 1) * H]
            future = self.predict_future(z, chunk, goal=goal, past_actions=past, **kwargs)
            z, past = torch.cat((z, future), dim=1), torch.cat((past, chunk), dim=1)
        return z
