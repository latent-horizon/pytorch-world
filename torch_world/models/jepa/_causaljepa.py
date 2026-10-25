import itertools

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_world.models.base import WorldModel
from torch_world.nn.transformer import MLP, Transformer


class SlotAttention(nn.Module):
    """Slot attention: slots compete for input features through a softmax over slots, then a GRU refines them.

    The last iteration starts from ``slots.detach() + init - init.detach()``, the straight-through gradient that lets
    learned initial slots act as queries (BO-QSA).

    Locatello et al., "Object-Centric Learning with Slot Attention", NeurIPS 2020.
    Paper: https://arxiv.org/abs/2006.15055
    Jia, Liu and Huang, "Improving Object-centric Learning with Query Optimization", ICLR 2023.
    Paper: https://arxiv.org/abs/2210.08990
    """

    def __init__(self, dim: int, iters: int = 3, hidden_dim: int | None = None):
        super().__init__()
        self.iters, self.scale = iters, dim**-0.5
        self.norm_in, self.norm_slots, self.norm_mlp = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.k, self.v = (nn.Linear(dim, dim, bias=False) for _ in range(3))
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = MLP([dim, hidden_dim or 2 * dim, dim], act=nn.ReLU)

    def forward(self, x: Tensor, init: Tensor) -> tuple[Tensor, Tensor]:
        """Features ``(N, P, D)`` and initial slots ``(N, S, D)`` to slots ``(N, S, D)`` and attention ``(N, S, P)``."""
        x = self.norm_in(x)
        k, v, slots = self.k(x), self.v(x), init
        for i in range(self.iters):
            if i == self.iters - 1:
                slots = slots.detach() + init - init.detach()
            attn = (self.q(self.norm_slots(slots)) @ k.transpose(1, 2) * self.scale).softmax(1) + 1e-8
            updates = (attn / attn.sum(-1, keepdim=True)) @ v
            slots = self.gru(updates.flatten(0, 1), slots.flatten(0, 1)).view_as(slots)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, attn


def slot_distances(x: Tensor, ref: Tensor) -> Tensor:
    """Cheapest one-to-one matching of slots ``x`` to ``ref`` (both ``(N, S, D)``), by trying every permutation:
    the mean squared distance ``(N,)`` and the slots of ``x`` reordered to match ``ref`` ``(N, S, D)``."""
    S = x.size(1)
    if S > 8:
        raise ValueError(f"matching tries all {S}! permutations; it is meant for at most 8 slots")
    perms = torch.tensor(list(itertools.permutations(range(S))), device=x.device)
    pairs = (ref[:, :, None] - x[:, None]).square().mean(-1)  # (N, S_ref, S_x)
    cost, best = pairs[:, torch.arange(S, device=x.device), perms].mean(-1).min(1)
    return cost, x[torch.arange(len(x), device=x.device)[:, None], perms[best]]


def _grid(h: int, w: int, device) -> Tensor:
    """``(h * w, 4)`` coordinates ``(y, x, 1 - y, 1 - x)`` in ``[0, 1]``, the soft position embedding's input."""
    y, x = torch.meshgrid(torch.linspace(0, 1, h, device=device), torch.linspace(0, 1, w, device=device), indexing="ij")
    return torch.stack([y, x, 1 - y, 1 - x], -1).flatten(0, 1)


class SlotEncoder(nn.Module):
    """Object-centric image encoder: a CNN, slot attention from learned queries and a spatial-broadcast decoder.

    ``forward`` maps images ``(N, C, H, W)`` to slots ``(N, num_slots, slot_dim)`` and videos ``(B, T, C, H, W)`` to
    ``(B, T, num_slots, slot_dim)``. Every frame starts from the learned queries. ``track`` keeps each object in the same
    slot through a video: ``"match"`` reorders the slots of frame ``t`` to best match frame ``t - 1``, and
    ``"propagate"`` starts frame ``t`` from the slots of frame ``t - 1``, as SAVi and VideoSAUR do.
    ``forward(pixels, reconstruct=True)`` instead returns the pixel reconstruction loss that trains it: each slot is
    broadcast on a grid, decoded to RGB and a mask logit, and the masks are softmaxed over slots.

    Watters et al., "Spatial Broadcast Decoder", 2019. Paper: https://arxiv.org/abs/1901.07017

    Args:
        img_size: side of the square input images, a multiple of 8.
        channels: width of the CNN and the decoder.
    """

    def __init__(
        self,
        img_size: int = 64,
        num_slots: int = 5,
        slot_dim: int = 64,
        channels: int = 64,
        iters: int = 3,
        in_channels: int = 3,
        track: str | None = "match",
    ):
        super().__init__()
        if img_size % 8:
            raise ValueError(f"img_size must be a multiple of 8, not {img_size}")
        if track not in (None, "match", "propagate"):
            raise ValueError(f"track must be None, 'match' or 'propagate', not {track!r}")
        self.img_size, self.num_slots, self.slot_dim, self.track = img_size, num_slots, slot_dim, track
        c = channels
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, c, 5, 2, 2), nn.ReLU(),
            *(m for _ in range(3) for m in (nn.Conv2d(c, c, 5, 1, 2), nn.ReLU())),
        )  # fmt: skip
        self.enc_pos = nn.Linear(4, c)
        self.enc_mlp = nn.Sequential(nn.LayerNorm(c), MLP([c, slot_dim, slot_dim], act=nn.ReLU))
        self.queries = nn.Parameter(torch.randn(1, num_slots, slot_dim) * slot_dim**-0.5)
        self.slot_attention = SlotAttention(slot_dim, iters)
        self.dec_pos = nn.Linear(4, slot_dim)
        self.decoder = nn.Sequential(
            *(m for i in range(3) for m in (nn.ConvTranspose2d(slot_dim if i == 0 else c, c, 5, 2, 2, 1), nn.ReLU())),
            nn.Conv2d(c, c, 5, 1, 2), nn.ReLU(), nn.Conv2d(c, in_channels + 1, 3, 1, 1),
        )  # fmt: skip

    def features(self, x: Tensor) -> Tensor:
        """Images ``(N, C, H, W)`` to position-aware features ``(N, H/2 * W/2, slot_dim)``."""
        h = self.cnn(x)
        h = h.flatten(2).transpose(1, 2) + self.enc_pos(_grid(*h.shape[-2:], x.device))
        return self.enc_mlp(h)

    def slots(self, x: Tensor, init: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Images ``(N, C, H, W)`` to slots ``(N, S, D)`` and feature attention ``(N, S, P)``, from ``init`` or the
        learned queries."""
        init = self.queries.expand(x.size(0), -1, -1) if init is None else init
        return self.slot_attention(self.features(x), init)

    def decode(self, slots: Tensor) -> tuple[Tensor, Tensor]:
        """Slots ``(N, S, D)`` to images ``(N, C, H, W)`` and slot masks ``(N, S, 1, H, W)`` that sum to one."""
        g = self.img_size // 8
        h = slots.flatten(0, 1)[:, :, None, None].expand(-1, -1, g, g)
        h = h + self.dec_pos(_grid(g, g, slots.device)).T.view(1, -1, g, g)
        out = self.decoder(h).unflatten(0, slots.shape[:2])
        rgb, masks = out[:, :, :-1], out[:, :, -1:].softmax(1)
        return (rgb * masks).sum(1), masks

    def forward(self, pixels: Tensor, reconstruct: bool = False) -> Tensor | dict[str, Tensor]:
        if reconstruct:
            x = pixels.flatten(0, pixels.dim() - 4)
            recon, masks = self.decode(self.slots(x)[0])
            loss = (recon - x).square().mean()
            return {"loss": loss, "recon_loss": loss, "recon": recon, "masks": masks}
        if pixels.dim() == 4:
            return self.slots(pixels)[0]
        if self.track != "propagate":
            out = list(self.slots(pixels.flatten(0, 1))[0].unflatten(0, pixels.shape[:2]).unbind(1))
            for t in range(1, len(out) if self.track == "match" else 0):
                out[t] = slot_distances(out[t], out[t - 1])[1]
            return torch.stack(out, 1)
        out = [self.slots(pixels[:, 0])[0]]
        for t in range(1, pixels.size(1)):
            out.append(self.slots(pixels[:, t], out[-1])[0])
        return torch.stack(out, 1)


class CausalJEPA(WorldModel):
    """Causal-JEPA (C-JEPA): a masked object-level JEPA world model, whose predictor must recover hidden objects from
    the others.

    A frozen object-centric encoder turns every frame into ``N`` slots. Each timestep becomes ``N`` object tokens plus
    an action token (and a proprioception token), and a bidirectional transformer with no positional encoding along
    the object axis reads ``history`` steps and predicts the next ``num_preds``. Training masks the history of up to
    ``max_masked`` random objects after the first step: their tokens become queries
    ``phi(z^i_{t0}) + m + e_tau`` built from the first-step slot (the identity anchor), a learned mask token and the
    time embedding, the same queries that stand for every future token. The predictor can only recover a masked object
    from the other objects and the actions, a latent intervention that makes interactions necessary to learn:

    ``loss = sum_{masked i, 0 < tau <= t} ||z^_tau^i - z_tau^i||^2 + sum_{i, tau > t} ||z^_tau^i - z_tau^i||^2``.

    At test time the history is fully visible. Planning minimizes the distance between the final predicted slots and
    the goal's slots, after matching them (Hungarian, as in the reference), with CEM.

    Nam et al., "Causal-JEPA: Learning World Models through Object-Level Latent Masking", 2026.
    Paper: https://arxiv.org/abs/2602.11389
    Code: https://github.com/galilai-group/cjepa

    Args:
        encoder: frozen module mapping videos ``(B, T, C, H, W)`` to slots ``(B, T, num_slots, slot_dim)``, such as a
            trained ``SlotEncoder`` (VideoSAUR on DINOv2 features in the paper). Its weights are frozen here.
        action_dim: dimension of one action, times the frameskip when actions are stacked.
        max_masked: each training sample masks a number of objects drawn uniformly from ``0..max_masked``.
        proprio_dim: adds a proprioception token per step, predicted for the future like the objects.
        match: match predicted and goal slots by the cheapest permutation in ``cost``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int,
        slot_dim: int = 128,
        history: int = 3,
        num_preds: int = 1,
        max_masked: int = 2,
        proprio_dim: int | None = None,
        depth: int = 6,
        num_heads: int = 16,
        head_dim: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        match: bool = True,
    ):
        super().__init__()
        self.history, self.num_preds, self.max_masked, self.match = history, num_preds, max_masked, match
        self.encoder = encoder.requires_grad_(False)
        self.action_encoder = nn.Linear(action_dim, slot_dim)
        self.proprio_encoder = nn.Linear(proprio_dim, slot_dim) if proprio_dim else None
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, slot_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.time_embed = nn.Parameter(torch.randn(1, history + num_preds, 1, slot_dim))
        self.anchor_proj = nn.Linear(slot_dim, slot_dim)
        self.predictor = Transformer(
            slot_dim, depth, num_heads, mlp_dim=mlp_dim, head_dim=head_dim, dropout=dropout
        )  # fmt: skip
        self.out_proj = nn.Linear(slot_dim, slot_dim)

    @torch.no_grad()
    def encode(self, pixels: Tensor) -> Tensor:
        """``(B, T, C, H, W)`` to frozen slots ``(B, T, num_slots, slot_dim)``."""
        return self.encoder(pixels)

    def tokens(self, z: Tensor, actions: Tensor, proprio: Tensor | None = None) -> Tensor:
        """Slots ``(B, T, S, D)``, actions ``(B, T, A)`` and proprioception ``(B, T, P)`` to nodes
        ``(B, T, S + 1 [+ 1], D)``: the objects, the proprioception, then the action."""
        extra = [] if self.proprio_encoder is None else [self.proprio_encoder(proprio)[:, :, None]]
        return torch.cat([z, *extra, self.action_encoder(actions)[:, :, None]], dim=2)

    def complete(self, x: Tensor, masked: Tensor | None = None, steps: int | None = None) -> Tensor:
        """Run the predictor on history nodes ``x`` ``(B, T, N, D)`` (``T <= history``) whose nodes ``masked``
        ``(B, N)`` are hidden after the first step, plus ``steps`` future steps (``num_preds`` by default) of queries;
        returns predictions for all ``(B, T + steps, N, D)``."""
        (B, T, N, D), P = x.shape, steps or self.num_preds
        pos = self.time_embed[:, self.history - T : self.history + P]
        query = self.mask_token + self.anchor_proj(x[:, :1])
        hist = x + pos[:, :T]
        if masked is not None:
            hidden = torch.cat([masked.new_zeros(B, 1, N), masked[:, None].expand(B, T - 1, N)], 1)
            hist = torch.where(hidden[..., None], query + pos[:, :T], hist)
        seq = torch.cat([hist, (query + pos[:, T:]).expand(B, P, N, D)], dim=1)
        return self.out_proj(self.predictor(seq.flatten(1, 2))).unflatten(1, (T + P, N))

    def predict(
        self, z: Tensor, actions: Tensor, proprio: Tensor | None = None, masked: Tensor | None = None
    ) -> Tensor:
        """From slots ``z`` ``(B, T, S, D)`` and their actions ``(B, T, A)``, the slots ``(B, T + num_preds, S, D)``:
        the history, where objects ``masked`` ``(B, S)`` are inferred after the first step, then the future."""
        proprio = None if proprio is None else proprio[:, -self.history :]
        x = self.tokens(z[:, -self.history :], actions[:, -self.history :], proprio)
        if masked is not None:
            masked = F.pad(masked, (0, x.size(2) - z.size(2)))
        return self.complete(x, masked)[:, :, : z.size(2)]

    def rollout(self, z: Tensor, actions: Tensor, proprio: Tensor | None = None) -> Tensor:
        """Unroll from ``T0`` context slots ``(B, T0, S, D)`` with ``T >= T0`` actions; returns ``(B, T + 1, S, D)``.

        Each step predicts the next frame (with its proprioception token) from the last ``history`` frames.
        ``proprio`` ``(B, T0, P)`` describes the context frames.
        """
        S, T0 = z.size(2), z.size(1)
        x = self.tokens(z, actions[:, :T0], proprio)
        for t in range(T0, actions.size(1) + 1):
            nxt = self.complete(x[:, -self.history :], steps=1)[:, -1]
            if t < actions.size(1):
                nxt = torch.cat([nxt[:, :-1], self.action_encoder(actions[:, t])[:, None]], dim=1)
            x = torch.cat([x, nxt[:, None]], dim=1)
        return torch.cat([z, x[:, T0:, :S]], dim=1)

    def cost(self, z: Tensor, goal: Tensor) -> Tensor:
        """Mean squared distance between the final slots of each rollout ``(B, T, S, D)`` and the goal's slots
        ``(B, S, D)``, under the cheapest one-to-one matching of slots when ``match``."""
        if not self.match:
            return (z[:, -1] - goal).square().flatten(1).mean(1)
        return slot_distances(z[:, -1], goal)[0]

    def forward(self, obs: Tensor, actions: Tensor, proprio: Tensor | None = None) -> dict[str, Tensor]:
        """``obs`` is pixels ``(B, history + num_preds, C, H, W)`` or pre-extracted slots ``(B, history + num_preds,
        S, D)``, with ``actions`` ``(B, history + num_preds, A)``."""
        z = obs if obs.dim() == 4 else self.encode(obs)
        (B, _, S, _), H = z.shape, self.history
        x = self.tokens(z, actions, proprio)
        count = torch.randint(0, self.max_masked + 1, (B, 1), device=z.device)
        masked = torch.rand(B, S, device=z.device).argsort(1).argsort(1) < count  # ``count`` random objects
        pred = self.complete(x[:, :H], F.pad(masked, (0, x.size(2) - S)))
        err = (pred[:, 1:H, :S] - z[:, 1:H]).square().mean(-1)  # (B, H - 1, S)
        hidden = masked[:, None].expand_as(err)
        history_loss = (err * hidden).sum() / hidden.sum().clamp(min=1)
        future_loss = (pred[:, H:, :S] - z[:, H:]).square().mean()
        out = {"history_loss": history_loss, "future_loss": future_loss}
        if self.proprio_encoder is not None:
            out["proprio_loss"] = (pred[:, H:, S] - x[:, H:, S].detach()).square().mean()
        return {"loss": sum(out.values()), **out}
