import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_world.models.base import WorldModel
from torch_world.nn.transformer import Transformer


class AdaJEPA(WorldModel):
    """AdaJEPA: a JEPA world model that keeps learning at test time, inside the planning loop.

    Each frame is encoded to one global latent, normalized by a LayerNorm. A causal transformer reads the last
    ``history`` latents, each concatenated with its action embedding, and predicts the next latents; the action
    features of its output are dropped. Training is teacher-forced one-step prediction against stop-gradient targets, with the temporal
    straightening (curvature) term of the base model and VCReg variance and covariance terms on the latents:

    ``loss = pred + curvature_coeff * curvature + std_coeff * std + cov_coeff * cov``

    At test time the model plans, runs the first action, and adds the observed transition ``(o_t, a_t, o_{t+1})`` to a
    buffer of the most recent ones. Before planning again it takes ``steps`` gradient steps of the same stop-gradient
    prediction loss on that buffer, updating only the last predictor block with the final norm and the encoder's
    last layer; every episode starts again from the trained weights. ``adapt`` does one such update and ``policy``
    builds the closed loop.

    Wang, Bounou, LeCun and Ren, "AdaJEPA: An Adaptive Latent World Model", 2026.
    Paper: https://arxiv.org/abs/2606.32026
    Code: https://github.com/agentic-learning-ai-lab/adajepa

    Args:
        encoder: module mapping images ``(N, C, H, W)`` to latents ``(N, embed_dim)``, such as ``PLDMEncoder``. Its
            last child module with parameters is the head adapted at test time.
        action_dim: dimension of one action.
        action_embed_dim: width of the action embedding concatenated to each latent (10 in the paper).
        curvature_coeff, std_coeff, cov_coeff: weights of the straightening, variance and covariance terms.
    """

    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int,
        embed_dim: int = 128,
        action_embed_dim: int = 16,
        history: int = 3,
        depth: int = 4,
        num_heads: int = 4,
        head_dim: int = 32,
        mlp_dim: int = 512,
        dropout: float = 0.0,
        curvature_coeff: float = 0.1,
        std_coeff: float = 1.0,
        cov_coeff: float = 0.04,
    ):
        super().__init__()
        self.history, self.embed_dim = history, embed_dim
        self.curvature_coeff, self.std_coeff, self.cov_coeff = curvature_coeff, std_coeff, cov_coeff
        self.encoder = encoder
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.action_encoder = nn.Linear(action_dim, action_embed_dim)
        dim = embed_dim + action_embed_dim
        self.pos_embed = nn.Parameter(torch.randn(1, history, dim) * 0.02)
        self.predictor = Transformer(dim, depth, num_heads, mlp_dim=mlp_dim, head_dim=head_dim, dropout=dropout)

    def encode(self, pixels: Tensor) -> Tensor:
        """``(B, T, C, H, W)`` to ``(B, T, embed_dim)``."""
        return self.norm(self.encoder(pixels.flatten(0, 1))).unflatten(0, pixels.shape[:2])

    def predict(self, z: Tensor, actions: Tensor) -> Tensor:
        T = z.size(1)
        x = torch.cat([z, self.action_encoder(actions)], -1) + self.pos_embed[:, :T]
        return self.predictor(x, is_causal=True)[..., : self.embed_dim]

    def prediction_loss(self, z: Tensor, actions: Tensor) -> Tensor:
        """One-step MSE against stop-gradient targets over every window of ``history`` steps of ``z`` ``(B, T + 1, D)``
        and ``actions`` ``(B, T, A)``."""
        w = min(self.history, actions.size(1))
        windows = range(actions.size(1) - w + 1)
        src = torch.cat([z[:, t : t + w] for t in windows])
        act = torch.cat([actions[:, t : t + w] for t in windows])
        tgt = torch.cat([z[:, t + 1 : t + 1 + w] for t in windows]).detach()
        return (self.predict(src, act) - tgt).square().mean()

    def forward(self, pixels: Tensor, actions: Tensor) -> dict[str, Tensor]:
        """``pixels`` ``(B, history + 1, C, H, W)`` and ``actions`` ``(B, history + 1, action_dim)``; the last action
        is unused."""
        z = self.encode(pixels)
        pred = self.prediction_loss(z[:, : self.history + 1], actions[:, : self.history])
        v1, v2 = z[:, 1:-1] - z[:, :-2], z[:, 2:] - z[:, 1:-1]
        curvature = (1 - F.cosine_similarity(v1, v2, dim=-1)).mean() if z.size(1) > 2 else z.new_zeros(())
        flat = z.flatten(0, 1)
        flat = flat - flat.mean(0)
        std = (1 - (flat.var(0) + 1e-4).sqrt()).relu().mean()
        cov = flat.T @ flat / (len(flat) - 1)
        cov = (cov.square().sum() - cov.diagonal().square().sum()) / flat.size(1)
        loss = pred + self.curvature_coeff * curvature + self.std_coeff * std + self.cov_coeff * cov
        return {"loss": loss, "pred_loss": pred, "curvature_loss": curvature, "std_loss": std, "cov_loss": cov}

    def adapted_parameters(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """The parameters updated at test time: the last predictor block with the final norm, and the encoder head."""
        predictor = [*self.predictor.blocks[-1].parameters(), *self.predictor.norm.parameters()]
        head = [m for m in self.encoder.children() if any(True for _ in m.parameters())]
        return predictor, list(head[-1].parameters() if head else self.encoder.parameters())

    def adapt(
        self, pixels: Tensor, actions: Tensor, steps: int = 1, lr: float = 5e-4, encoder_lr: float = 1e-5
    ) -> list[float]:
        """Take ``steps`` Adam steps of the stop-gradient prediction loss on observed transitions, in place.

        ``pixels`` ``(B, T + 1, C, H, W)`` and ``actions`` ``(B, T, action_dim)`` hold the frames of the replay buffer
        and the actions between them. A fresh optimizer is made on every call, as in the reference; ``encoder_lr=0``
        keeps the encoder frozen. Returns the loss before each step.
        """
        predictor, head = self.adapted_parameters()
        groups = [{"params": predictor, "lr": lr}] + ([{"params": head, "lr": encoder_lr}] if encoder_lr else [])
        trainable = [p for g in groups for p in g["params"]]
        saved = [p.requires_grad for p in self.parameters()]
        self.requires_grad_(False)
        for p in trainable:
            p.requires_grad_(True)
        opt, losses = torch.optim.Adam(groups), []
        with torch.enable_grad():
            if not encoder_lr:
                with torch.no_grad():
                    z = self.encode(pixels)
            for _ in range(steps):
                loss = self.prediction_loss(self.encode(pixels) if encoder_lr else z, actions)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
        for p, r in zip(self.parameters(), saved):
            p.requires_grad_(r)
        return losses

    def policy(
        self,
        planner,
        goal_images: Tensor,
        buffer: int = 5,
        steps: int = 1,
        lr: float = 5e-4,
        encoder_lr: float = 1e-5,
    ) -> "AdaptivePolicy":
        """Closed-loop policy that adapts the model on the last ``buffer`` transitions of each episode before every
        plan; ``steps=0`` gives the same loop with the model frozen. See ``AdaptivePolicy``."""
        return AdaptivePolicy(self, planner, goal_images, buffer, steps, lr, encoder_lr)


class AdaptivePolicy:
    """AdaJEPA's plan-and-adapt loop as a ``policy(observation) -> action`` for ``evaluation.goal_reaching``.

    Each episode in the batch keeps its own copy of the adapted parameters and its own buffer of recent transitions,
    starting from the trained weights. On every call, for each episode: the new observation completes the last
    transition, the model is adapted on the buffer, observation and goal image are encoded with the adapted encoder,
    ``planner(model, z, goal)`` plans, and the first action runs. Episodes are planned one at a time; the model's
    weights are restored after every call. Call ``reset`` between episodes.
    """

    def __init__(self, model: AdaJEPA, planner, goal_images: Tensor, buffer=5, steps=1, lr=5e-4, encoder_lr=1e-5):
        self.model, self.planner, self.goal_images = model, planner, goal_images
        self.buffer, self.steps, self.lr, self.encoder_lr = buffer, steps, lr, encoder_lr
        self.params = [p for group in model.adapted_parameters() for p in group]
        self.losses: list[list[float]] = []
        self.reset()

    def reset(self):
        self.initial = [p.detach().clone() for p in self.params]
        self.state: dict[int, list[Tensor]] = {}
        self.frames: dict[int, list[Tensor]] = {}
        self.actions: dict[int, list[Tensor]] = {}
        self.losses = []

    @torch.no_grad()
    def _load(self, tensors: list[Tensor]):
        for p, t in zip(self.params, tensors):
            p.copy_(t)

    def __call__(self, observation: Tensor) -> Tensor:
        device = next(self.model.parameters()).device
        out, step_losses = [], []
        for i, obs in enumerate(observation.to(device)):
            self._load(self.state.get(i, self.initial))
            frames, actions = self.frames.setdefault(i, []), self.actions.setdefault(i, [])
            frames.append(obs)
            del frames[: -self.buffer - 1], actions[: -self.buffer]
            if actions and self.steps:
                pixels, acts = torch.stack(frames[-len(actions) - 1 :])[None], torch.stack(actions)[None]
                step_losses.append(self.model.adapt(pixels, acts, self.steps, self.lr, self.encoder_lr)[0])
            with torch.no_grad():
                z = self.model.encode(obs[None, None])
                goal = self.model.encode(self.goal_images[i : i + 1, None].to(device))[:, 0]
                action = self.planner(self.model, z, goal)[0, 0]
            actions.append(action)
            out.append(action)
            self.state[i] = [p.detach().clone() for p in self.params]
        self._load(self.initial)
        if step_losses:
            self.losses.append(step_losses)
        return torch.stack(out).to(observation.device)
