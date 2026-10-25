import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_world.models.base import WorldActionModel
from torch_world.nn.flow import euler_sample, interpolate, sample_time, time_embedding, velocity_target
from torch_world.nn.positional import sincos_1d
from torch_world.nn.transformer import MLP, Transformer


class VLAJEPA(WorldActionModel):
    """VLA-JEPA: a vision-language-action policy whose VLM also drives a latent world model, trained JEPA-style.

    A causal VLM reads the current image, the instruction, then two groups of learnable query tokens: one
    *latent-action* token per transition (each repeated ``latent_tokens`` times) and an *embodied-action* token
    (repeated ``action_tokens`` times). A frozen target encoder embeds every frame of the clip into world states
    ``s_0 .. s_{T-1}``; future frames are only regression targets and never reach the VLM. An action-conditioned
    predictor with frame-causal attention reads the true states ``s_{<T-1}`` with the latent-action tokens interleaved
    per step and predicts ``s_{1:T}`` (teacher forcing, L1). A flow-matching action expert, conditioned on the
    embodied-action tokens and on the flow time through adaLN-zero, denoises the next ``chunk_size`` actions:

    ``loss = || v(a_t, t | e) - (a - eps) ||^2 + wm_weight * | P(s_{<T-1}, l) - s_{1:T} |``, ``a_t = (1 - t) eps + t a``.

    Clips without actions (human video) train the world-model term alone. Inference runs only the VLM and the expert:
    ``sample_actions`` integrates the flow from noise in ``flow_steps`` Euler steps. ``imagine`` also unrolls the world
    model from the current state with the latent-action tokens, the futures the VLM expects under the instruction.

    The world model is conditioned on the VLM's latent-action tokens, not on robot actions, so this is a
    ``WorldActionModel`` (it outputs actions and predicts their consequences) whose ``predict`` and ``rollout`` take
    latent-action tokens ``(B, T, latent_tokens, dim)`` in place of actions; action-space planners such as ``CEM`` do
    not apply. The paper's L_WM weight is not given; ``wm_weight = 0.1`` is the released code's.

    Sun et al., "VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model", 2026.
    Paper: https://arxiv.org/abs/2602.10098
    Code: https://github.com/ginwind/VLA-JEPA

    Args:
        vision: trainable module mapping images ``(N, C, H, W)`` to tokens ``(N, P, vision_dim)`` or ``(N, vision_dim)``
            (Qwen3-VL's SigLIP-2 in the paper).
        target_encoder: frozen module mapping images to world-state tokens ``(N, num_patches, target_dim)`` or
            ``(N, target_dim)`` (V-JEPA 2 ViT-L in the paper).
        vocab_size: size of the instruction vocabulary; id 0 is padding (``WordTokenizer``).
        frames: world states per clip ``T``, so ``T - 1`` latent-action groups (8 in the paper).
        latent_tokens: copies of each latent-action token (``K = 24 / T = 3`` in the paper, 8 in the released config).
        action_tokens: copies of the embodied-action token that conditions the expert (32).
        flow_repeats: noise draws per sample in the flow-matching loss (``repeated_diffusion_steps``, 8).
        time_alpha: flow times are ``1 - Beta(time_alpha, 1)``, concentrated near noise, as GR00T's head draws them.
    """

    def __init__(
        self,
        vision: nn.Module,
        target_encoder: nn.Module,
        vocab_size: int,
        action_dim: int,
        vision_dim: int,
        target_dim: int,
        num_patches: int = 1,
        *,
        frames: int = 8,
        chunk_size: int = 7,
        latent_tokens: int = 3,
        action_tokens: int = 32,
        state_dim: int = 0,
        dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        predictor_dim: int = 256,
        predictor_depth: int = 4,
        predictor_heads: int = 4,
        expert_dim: int = 256,
        expert_depth: int = 4,
        expert_heads: int = 4,
        flow_steps: int = 4,
        flow_repeats: int = 8,
        time_alpha: float = 1.5,
        wm_weight: float = 0.1,
    ):
        super().__init__()
        if frames < 2:
            raise ValueError("frames must be at least 2")
        self.frames, self.chunk_size, self.action_dim, self.state_dim = frames, chunk_size, action_dim, state_dim
        self.latent_tokens, self.action_tokens, self.dim = latent_tokens, action_tokens, dim
        self.flow_steps, self.flow_repeats, self.time_alpha, self.wm_weight = (
            flow_steps,
            flow_repeats,
            time_alpha,
            wm_weight,
        )
        # VLM: vision tokens, instruction words, latent-action queries, embodied-action queries, causal attention
        self.vision, self.vision_proj = vision, nn.Linear(vision_dim, dim)
        self.word_embed = nn.Embedding(vocab_size, dim)
        self.latent_query = nn.Parameter(torch.randn(frames - 1, dim) * 0.02)
        self.action_query = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.vlm = Transformer(dim, depth, num_heads)
        # world model: frozen target encoder and a frame-causal predictor over [latent-action tokens, state tokens]
        self.target_encoder = target_encoder.requires_grad_(False).eval()
        self.state_embed, self.latent_embed = nn.Linear(target_dim, predictor_dim), nn.Linear(dim, predictor_dim)
        self.patch_pos = nn.Parameter(torch.randn(1, 1, num_patches, predictor_dim) * 0.02)
        self.predictor = Transformer(predictor_dim, predictor_depth, predictor_heads)
        self.predictor_proj = nn.Linear(predictor_dim, target_dim)
        # action expert: flow matching on action chunks, conditioned on the embodied-action tokens
        self.cond_proj = nn.Linear(dim, expert_dim)
        self.action_in = MLP([action_dim, expert_dim, expert_dim], act=nn.SiLU)
        self.action_pos = nn.Parameter(torch.randn(1, chunk_size, expert_dim) * 0.02)
        self.state_in = MLP([state_dim, expert_dim, expert_dim], act=nn.SiLU) if state_dim else None
        self.time_mlp = MLP([expert_dim, expert_dim, expert_dim], act=nn.SiLU)
        self.expert = Transformer(expert_dim, expert_depth, expert_heads, conditional=True)
        self.action_out = nn.Linear(expert_dim, action_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def encode(self, pixels: Tensor) -> Tensor:
        """World states of frames ``(B, T, C, H, W)`` from the frozen target encoder: ``(B, T, P, target_dim)``."""
        z = self.target_encoder(pixels.flatten(0, 1))
        return (z if z.dim() == 3 else z[:, None]).unflatten(0, pixels.shape[:2]).float()

    def context(self, image: Tensor, ids: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Run the VLM on images ``(B, C, H, W)`` and instructions ``ids`` ``(B, L)`` (``mask`` is ``True`` on words).

        Returns the latent-action tokens ``(B, frames - 1, latent_tokens, dim)`` and the embodied-action tokens
        ``(B, action_tokens, dim)``.
        """
        v = self.vision(image)
        v = self.vision_proj(v if v.dim() == 3 else v[:, None])
        B, n = len(v), (self.frames - 1) * self.latent_tokens
        queries = torch.cat(
            [
                self.latent_query.repeat_interleave(self.latent_tokens, 0),
                self.action_query.expand(self.action_tokens, -1),
            ]
        )
        x = torch.cat([v, self.word_embed(ids), queries.expand(B, -1, -1)], dim=1)
        x = x + sincos_1d(self.dim, torch.arange(x.size(1))).to(x)
        mask = torch.ones_like(ids, dtype=torch.bool) if mask is None else mask.bool()
        valid = torch.cat([mask.new_ones(B, v.size(1)), mask, mask.new_ones(B, len(queries))], dim=1)
        causal = torch.ones(x.size(1), x.size(1), dtype=torch.bool, device=x.device).tril()
        h = self.vlm(x, attn_mask=(causal & valid[:, None, :])[:, None])
        return h[:, -n - self.action_tokens : -self.action_tokens].unflatten(1, (-1, self.latent_tokens)), h[
            :, -self.action_tokens :
        ]

    def predict(self, z: Tensor, actions: Tensor) -> Tensor:
        """Next world states ``(B, T, P, D)`` from states ``z`` ``(B, T, P, D)`` and latent-action tokens ``actions``
        ``(B, T, latent_tokens, dim)``; step ``t`` sees the tokens of steps ``<= t`` only."""
        T, P, K = z.size(1), z.size(2), actions.size(2)
        x = torch.cat([self.latent_embed(actions), self.state_embed(z) + self.patch_pos[:, :, :P]], dim=2)
        x = x + sincos_1d(x.size(-1), torch.arange(T))[None, :, None].to(x)
        step = torch.arange(T, device=x.device).repeat_interleave(K + P)
        h = self.predictor(x.flatten(1, 2), attn_mask=step[None, :] <= step[:, None])
        return self.predictor_proj(h.unflatten(1, (T, K + P))[:, :, K:])

    def predict_future(self, z: Tensor, actions: Tensor, goal: Tensor | None = None) -> Tensor:
        """Unroll the world model from ``z[:, :1]`` with latent-action tokens ``(B, T, K, dim)``: ``(B, T, P, D)``."""
        return self.rollout(z[:, :1], actions)[:, 1:]

    def velocity(self, x: Tensor, t: Tensor, cond: Tensor, state: Tensor | None = None) -> Tensor:
        """Action velocity ``(B, H, action_dim)`` at noisy chunks ``x`` and flow times ``t`` ``(B,)``, given the
        embodied-action tokens ``cond`` ``(B, M, dim)`` and optional states ``(B, state_dim)``."""
        tokens = [self.cond_proj(cond)]
        if self.state_in is not None:
            tokens.append(self.state_in(state)[:, None])
        tokens.append(self.action_in(x) + self.action_pos[:, : x.size(1)])
        c = self.time_mlp(time_embedding(t, self.time_mlp[0].in_features).to(x))[:, None]
        return self.action_out(self.expert(torch.cat(tokens, dim=1), c)[:, -x.size(1) :])

    def _sample(self, cond, state, steps, generator):
        shape = (len(cond), self.chunk_size, self.action_dim)
        noise = torch.randn(shape, device=cond.device, dtype=cond.dtype, generator=generator)
        return euler_sample(lambda x, t: self.velocity(x, t, cond, state), noise, steps or self.flow_steps)

    def sample_actions(
        self,
        image: Tensor,
        ids: Tensor,
        mask: Tensor | None = None,
        state: Tensor | None = None,
        steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Action chunk ``(B, chunk_size, action_dim)`` for images ``(B, C, H, W)`` and instructions ``(B, L)``."""
        return self._sample(self.context(image, ids, mask)[1], state, steps, generator)

    def imagine(
        self, image: Tensor, ids: Tensor, mask: Tensor | None = None, state: Tensor | None = None, **kwargs
    ) -> tuple[Tensor, Tensor]:
        """An action chunk and the world states the VLM expects under the instruction, ``(B, frames, P, D)``."""
        latent, cond = self.context(image, ids, mask)
        return self._sample(cond, state, kwargs.get("steps"), kwargs.get("generator")), self.rollout(
            self.encode(image[:, None]), latent
        )

    def forward(
        self,
        pixels: Tensor,
        ids: Tensor,
        mask: Tensor | None = None,
        actions: Tensor | None = None,
        state: Tensor | None = None,
        latents: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """``pixels`` ``(B, T, C, H, W)`` with ``T <= frames``: the VLM sees ``pixels[:, 0]`` and the target encoder
        every frame, unless precomputed world states ``latents`` ``(B, T, P, D)`` are given (then ``pixels`` may hold
        the first frame only). ``actions`` ``(B, chunk_size, action_dim)`` are the chunk from the first frame; without
        them only the world-model loss is trained, as for human videos."""
        z = self.encode(pixels) if latents is None else latents.float()
        latent, cond = self.context(pixels[:, 0], ids, mask)
        wm = F.l1_loss(self.predict(z[:, :-1], latent[:, : z.size(1) - 1]), z[:, 1:])
        if actions is None:
            return {"loss": wm, "wm_loss": wm}
        r = self.flow_repeats
        a, cond = actions.repeat(r, 1, 1), cond.repeat(r, 1, 1)
        state = None if state is None else state.repeat(r, 1)
        noise = torch.randn_like(a)
        t = sample_time(len(a), device=a.device, dtype=a.dtype, alpha=self.time_alpha)
        fm = (self.velocity(interpolate(noise, a, t), t, cond, state) - velocity_target(noise, a)).square().mean()
        return {"loss": fm + self.wm_weight * wm, "action_loss": fm, "wm_loss": wm}
