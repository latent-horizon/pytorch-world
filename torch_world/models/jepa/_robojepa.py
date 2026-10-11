import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_world.masks.utils import frame_causal_mask
from torch_world.models.base import WorldModel
from torch_world.nn.transformer import Transformer, init_weights, rescale_blocks


class EmbodimentLinear(nn.Module):
    """Linear map with separate weights per embodiment, selected by ``ids`` ``(B,)``."""

    def __init__(self, in_dim: int, out_dim: int, num_embodiments: int = 1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embodiments, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(num_embodiments, out_dim))
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, x: Tensor, ids: Tensor) -> Tensor:
        return torch.einsum("bti,bio->bto", x, self.weight[ids]) + self.bias[ids, None]


class RoboJEPA(WorldModel):
    """RoboJEPA: an action-conditioned predictor in the latent space of a frozen V-JEPA 2.1 encoder, for many robots.

    Every timestep contributes the patch tokens of all camera views, one action token and one proprioceptive state
    token. Actions and states are projected per embodiment. Attention is causal over frames within a ``window`` of
    past frames, and positions use 3D RoPE ``(t, h, w)``. The loss is L1 on layer-normalized features, under
    teacher forcing and over ``rollout_steps`` autoregressive steps, as in V-JEPA 2-AC.

    Zholus et al., "RoboJEPA: Scaling Robotic Latent World Models", 2026.
    Paper: https://arxiv.org/abs/2610.10515
    Code: https://github.com/facebookresearch/robo_jepa

    Assran et al., "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning", 2025.
    Paper: https://arxiv.org/abs/2506.09985
    Code: https://github.com/facebookresearch/vjepa2

    Latents are ``(B, T, V, H, W, embed_dim)``: ``V`` views on an ``H x W`` patch grid.
    """

    def __init__(
        self,
        embed_dim: int,
        action_dim: int,
        state_dim: int = 0,
        num_views: int = 1,
        num_embodiments: int = 1,
        predictor_dim: int = 512,
        depth: int = 7,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        window: int | None = 8,
        rollout_steps: int = 2,
        encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.window = window
        self.rollout_steps = rollout_steps
        self.encoder = encoder.requires_grad_(False).eval() if encoder is not None else None
        self.embed = nn.Linear(embed_dim, predictor_dim)
        self.view_embed = nn.Parameter(torch.zeros(num_views, predictor_dim))
        self.embodiment_embed = nn.Embedding(num_embodiments, predictor_dim)
        self.action_proj = EmbodimentLinear(action_dim, predictor_dim, num_embodiments)
        self.state_proj = EmbodimentLinear(state_dim, predictor_dim, num_embodiments) if state_dim else None
        self.predictor = Transformer(
            predictor_dim, depth, num_heads, mlp_dim=int(predictor_dim * mlp_ratio), use_rope=True
        )
        self.proj = nn.Linear(predictor_dim, embed_dim)
        init_weights(self.predictor)
        rescale_blocks(self.predictor.blocks)
        nn.init.trunc_normal_(self.view_embed, std=0.02)

    @property
    def num_cond(self) -> int:
        return 1 + (self.state_proj is not None)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.encoder is not None:
            self.encoder.eval()
        return self

    @torch.no_grad()
    def encode(self, frames: Tensor, grid: tuple[int, int]) -> Tensor:
        """``frames`` ``(B, T, V, C, H, W)`` to layer-normalized features ``(B, T, V, h, w, D)`` with the frozen encoder."""
        z = self.encoder(frames.flatten(0, 2)).unflatten(1, grid).unflatten(0, frames.shape[:3])
        return F.layer_norm(z, z.shape[-1:])

    def _positions(self, T: int, V: int, H: int, W: int, device) -> Tensor:
        t, h, w = torch.meshgrid(torch.arange(T), torch.arange(H), torch.arange(W), indexing="ij")
        patches = torch.stack([t, h, w], dim=-1)[:, None].expand(T, V, H, W, 3).reshape(T, V * H * W, 3)
        cond = torch.stack(
            [torch.arange(T), torch.zeros(T, dtype=torch.long), torch.zeros(T, dtype=torch.long)], dim=-1
        )
        cond = cond[:, None].expand(T, self.num_cond, 3)
        return torch.cat([patches, cond], dim=1).flatten(0, 1).float().to(device)

    def predict(
        self, z: Tensor, actions: Tensor, states: Tensor | None = None, embodiment: Tensor | None = None
    ) -> Tensor:
        B, T, V, H, W, _ = z.shape
        ids = embodiment if embodiment is not None else torch.zeros(B, dtype=torch.long, device=z.device)
        x = self.embed(z) + self.view_embed[:V, None, None] + self.embodiment_embed(ids)[:, None, None, None, None]
        cond = [self.action_proj(actions, ids)]
        if self.state_proj is not None:
            cond.append(self.state_proj(states, ids))
        x = torch.cat([x.flatten(2, 4), torch.stack(cond, dim=2)], dim=2)
        n = x.size(2)
        mask = frame_causal_mask(T, n, self.window, device=z.device)
        x = self.predictor(x.flatten(1, 2), attn_mask=mask, pos=self._positions(T, V, H, W, z.device))
        return self.proj(x.unflatten(1, (T, n))[:, :, : V * H * W]).unflatten(2, (V, H, W))

    def rollout(
        self, z: Tensor, actions: Tensor, states: Tensor | None = None, embodiment: Tensor | None = None
    ) -> Tensor:
        """States past the given ones repeat the last known state."""
        if states is not None and states.size(1) < actions.size(1):
            pad = states[:, -1:].expand(-1, actions.size(1) - states.size(1), -1)
            states = torch.cat([states, pad], dim=1)
        for t in range(z.size(1), actions.size(1) + 1):
            s = None if states is None else states[:, :t]
            z = torch.cat([z, self.predict(z, actions[:, :t], s, embodiment)[:, -1:]], dim=1)
        return z

    def cost(self, z: Tensor, goal: Tensor) -> Tensor:
        return (z[:, -1] - goal).abs().flatten(1).mean(1)

    def forward(
        self, z: Tensor, actions: Tensor, states: Tensor | None = None, embodiment: Tensor | None = None
    ) -> dict[str, Tensor]:
        """``z`` ``(B, T, V, H, W, D)`` encoder features, ``actions`` ``(B, T, A)`` and ``states`` ``(B, T, S)``."""
        z = F.layer_norm(z, z.shape[-1:])
        tf = self.predict(z[:, :-1], actions[:, :-1], None if states is None else states[:, :-1], embodiment)
        tf_loss = (tf - z[:, 1:]).abs().mean()
        ar = torch.cat([z[:, :1], tf[:, :1]], dim=1)
        for n in range(1, self.rollout_steps):
            s = None if states is None else states[:, : n + 1]
            ar = torch.cat([ar, self.predict(ar, actions[:, : n + 1], s, embodiment)[:, -1:]], dim=1)
        ar_loss = (ar[:, 1:] - z[:, 1 : self.rollout_steps + 1]).abs().mean()
        return {"loss": tf_loss + ar_loss, "tf_loss": tf_loss, "rollout_loss": ar_loss}
