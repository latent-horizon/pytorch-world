import math
from collections.abc import Callable
from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call

from torch_world.models.base import JEPA
from torch_world.models.tdmpc._tdmpc2 import ShiftAug, _Center
from torch_world.utils.ema import ema_update


def _mlp(in_dim: int, hidden_dim: int, hidden_layers: int, out_dim: int) -> nn.Sequential:
    """``hidden_layers`` Linear and ReLU layers, then a Linear to ``out_dim``."""
    dims = [in_dim] + [hidden_dim] * hidden_layers
    layers = [m for a, b in pairwise(dims) for m in (nn.Linear(a, b), nn.ReLU())]
    return nn.Sequential(*layers, nn.Linear(dims[-1], out_dim))


def _embedding(in_dim: int, hidden_dim: int) -> nn.Sequential:
    """The two embedding layers of Touati et al.'s networks: Linear, LayerNorm and Tanh, then Linear and ReLU to half
    the width."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
    )


def sphere(x: Tensor) -> Tensor:
    """Project ``(..., d)`` onto the sphere of radius ``sqrt(d)``."""
    return math.sqrt(x.size(-1)) * F.normalize(x, dim=-1)


class _TrainShift(ShiftAug):
    """``ShiftAug`` in training mode only, as DrQ-v2 augments."""

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x) if self.training else x


def drq_encoder(
    out_dim: int, hidden_layers: int = 0, hidden_dim: int = 256, img_size: int = 64, channels: int = 3, pad: int = 2
) -> nn.Sequential:
    """TD-JEPA's pixel encoder: DrQ-v2's convolutions (one 3x3 at stride 2, three at stride 1, 32 channels, ReLU) on
    images ``(N, C, H, W)`` in ``[0, 1]``, random shifts of up to ``pad`` pixels in training, then an MLP with
    ``hidden_layers`` ReLU layers to ``(N, out_dim)``. ``hidden_layers=0`` is the paper's pixel state encoder ``phi``
    (a linear map), ``hidden_layers=2`` its task encoder ``psi``."""
    side = (img_size - 3) // 2 + 1 - 6
    convs = [nn.Conv2d(channels, 32, 3, stride=2), nn.ReLU()]
    for _ in range(3):
        convs += [nn.Conv2d(32, 32, 3), nn.ReLU()]
    return nn.Sequential(
        _TrainShift(pad), _Center(), *convs, nn.Flatten(), _mlp(32 * side * side, hidden_dim, hidden_layers, out_dim)
    )


class TDPredictor(nn.Module):
    """Policy-conditioned predictor ``T(x, a, z)``: twin heads on two embeddings, one of ``[x, a]`` and one of
    ``[x, z]``, as Touati et al.'s forward map. Returns ``(num_twins, ..., out_dim)``."""

    def __init__(
        self,
        in_dim: int,
        action_dim: int,
        task_dim: int,
        out_dim: int,
        hidden_dim: int = 1024,
        hidden_layers: int = 1,
        num_twins: int = 2,
    ):
        super().__init__()
        self.embed_action = _embedding(in_dim + action_dim, hidden_dim)
        self.embed_task = _embedding(in_dim + task_dim, hidden_dim)
        self.heads = nn.ModuleList(_mlp(hidden_dim, hidden_dim, hidden_layers, out_dim) for _ in range(num_twins))

    def forward(self, x: Tensor, a: Tensor, z: Tensor) -> Tensor:
        h = torch.cat([self.embed_action(torch.cat([x, a], -1)), self.embed_task(torch.cat([x, z], -1))], -1)
        return torch.stack([head(h) for head in self.heads])


class TDActor(nn.Module):
    """Task-conditioned actor ``pi(x, z)``: an MLP on embeddings of ``[x, z]`` and ``x`` to the tanh mean ``(...,
    action_dim)`` of a Gaussian with fixed std, as Touati et al.'s actor."""

    def __init__(self, in_dim: int, action_dim: int, task_dim: int, hidden_dim: int = 256, hidden_layers: int = 1):
        super().__init__()
        self.embed_task = _embedding(in_dim + task_dim, hidden_dim)
        self.embed_state = _embedding(in_dim, hidden_dim)
        self.net = _mlp(hidden_dim, hidden_dim, hidden_layers, action_dim)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        return self.net(torch.cat([self.embed_task(torch.cat([x, z], -1)), self.embed_state(x)], -1)).tanh()


class TDJEPA(JEPA):
    """TD-JEPA: latent-predictive representations for zero-shot RL, learned from reward-free transitions with
    temporal-difference latent prediction.

    A state encoder ``phi`` and a task encoder ``psi`` embed observations. A predictor ``T_phi(phi(s), a, z)``
    predicts the discounted sum of future ``psi`` embeddings (the successor features of ``psi``) under the policy
    ``pi_z``, and ``T_psi(psi(s), a, z)`` symmetrically those of ``phi``. Both are trained by TD against EMA target
    networks, with the next action drawn from the policy, plus an orthonormality regularizer on each encoder:

    ``L = 1/2 || T_phi(phi(s), a, z) - psi-(s') - gamma T_phi-(phi-(s'), a', z) ||^2 + (same with phi, psi swapped)
    + lamb * (reg(phi) + reg(psi))``, with ``a' ~ pi(phi-(s'), z)`` and
    ``reg(phi) = mean_{i != j} (phi_i . phi_j)^2 / 2 - mean_i phi_i . phi_i``.

    Since ``T_phi(phi(s), a, z) . z`` is then the Q-value of ``pi_z`` for the reward ``r(s) = psi(s) . z``, the actor
    maximizes it for every ``z`` (through the predictor, which it cannot update). Task embeddings ``z`` are drawn on
    the sphere of radius ``sqrt(task_dim)``, or with probability ``p_goal`` are ``psi`` of random dataset states. At
    test time, no training is needed: ``infer_task`` regresses a reward onto ``psi``, ``goal_task`` embeds goal
    observations, and ``policy`` acts for the inferred ``z``.

    ``forward`` returns the encoder/predictor losses plus the actor loss; their gradients reach disjoint parameters,
    so one backward pass matches Algorithm 1's three updates. Call ``update_target`` after every optimizer step.

    Bagatella, Pirotta, Touati, Lazaric and Tirinzoni, "TD-JEPA: Latent-predictive Representations for Zero-Shot
    Reinforcement Learning", ICLR 2026.
    Paper: https://arxiv.org/abs/2510.00739
    Code: not released; this follows the paper.

    Args:
        phi_encoder: module mapping observations ``(N, ...)`` to ``(N, phi_dim)``, such as ``drq_encoder``.
        psi_encoder: module mapping observations ``(N, ...)`` to ``(N, task_dim)``.
        hidden_dim, hidden_layers: width and ReLU layers after the two embedding layers of the predictors.
        actor_dim, actor_layers: the same for the actor.
        num_twins: predictor heads; targets and values use their mean.
        lamb: weight of the orthonormality regularizer (``lambda``).
        p_goal: probability that a training ``z`` is ``psi`` of a random dataset state.
        actor_std, actor_clip: fixed std of the actor's Gaussian and the clip of its noise.
        normalize: project encoder outputs onto the sphere of radius ``sqrt(dim)`` (Touati et al.'s L2 normalization).
    """

    def __init__(
        self,
        phi_encoder: nn.Module,
        psi_encoder: nn.Module,
        action_dim: int,
        phi_dim: int = 256,
        task_dim: int = 50,
        hidden_dim: int = 1024,
        hidden_layers: int = 1,
        actor_dim: int = 256,
        actor_layers: int = 1,
        num_twins: int = 2,
        discount: float = 0.98,
        lamb: float = 1.0,
        p_goal: float = 0.5,
        actor_std: float = 0.2,
        actor_clip: float = 0.3,
        normalize: bool = True,
    ):
        super().__init__()
        self.action_dim, self.phi_dim, self.task_dim = action_dim, phi_dim, task_dim
        self.discount, self.lamb, self.p_goal = discount, lamb, p_goal
        self.actor_std, self.actor_clip, self.normalize = actor_std, actor_clip, normalize
        self.phi_encoder, self.psi_encoder = phi_encoder, psi_encoder
        self.phi_predictor = TDPredictor(phi_dim, action_dim, task_dim, task_dim, hidden_dim, hidden_layers, num_twins)
        self.psi_predictor = TDPredictor(task_dim, action_dim, task_dim, phi_dim, hidden_dim, hidden_layers, num_twins)
        self.actor = TDActor(phi_dim, action_dim, task_dim, actor_dim, actor_layers)
        self.targets = nn.ModuleDict(
            {
                name: self.make_target_encoder(getattr(self, name))
                for name in ("phi_encoder", "psi_encoder", "phi_predictor", "psi_predictor")
            }
        )

    def param_groups(self, lr: float = 1e-4, task_lr: float | None = None) -> list[dict]:
        """Adam groups: everything at ``lr``, the task encoder ``psi`` at ``task_lr`` (the paper sweeps 1e-4, 1e-5)."""
        psi = list(self.psi_encoder.parameters())
        skip = {id(p) for p in psi}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in skip]
        return [{"params": rest, "lr": lr}, {"params": psi, "lr": task_lr or lr}]

    def update_target(self, momentum: float = 0.995):
        """EMA of ``phi``, ``psi``, ``T_phi`` and ``T_psi``; the paper's EMA coefficient is ``1 - momentum``."""
        for name, target in self.targets.items():
            ema_update(target, getattr(self, name), momentum)

    def _embed(self, net: nn.Module, x: Tensor) -> Tensor:
        x = net(x)
        return sphere(x) if self.normalize else x

    def phi(self, obs: Tensor, target: bool = False) -> Tensor:
        """State embeddings ``(N, phi_dim)`` of observations ``(N, ...)``."""
        return self._embed(self.targets["phi_encoder"] if target else self.phi_encoder, obs)

    def psi(self, obs: Tensor, target: bool = False) -> Tensor:
        """Task embeddings ``(N, task_dim)`` of observations ``(N, ...)``."""
        return self._embed(self.targets["psi_encoder"] if target else self.psi_encoder, obs)

    def encode(self, obs: Tensor) -> Tensor:
        """``(B, T, ...)`` to state embeddings ``(B, T, phi_dim)``."""
        return self.phi(obs.flatten(0, 1)).unflatten(0, obs.shape[:2])

    def encode_task(self, obs: Tensor) -> Tensor:
        """``(B, T, ...)`` to task embeddings ``(B, T, task_dim)``."""
        return self.psi(obs.flatten(0, 1)).unflatten(0, obs.shape[:2])

    def sample_tasks(self, n: int, goals: Tensor | None = None) -> Tensor:
        """``n`` task embeddings ``(n, task_dim)``: uniform on the sphere of radius ``sqrt(task_dim)``, or with
        probability ``p_goal`` the projected ``psi`` of ``goals`` ``(n, ...)``."""
        device = next(self.parameters()).device
        z = sphere(torch.randn(n, self.task_dim, device=device))
        if goals is not None and self.p_goal > 0:
            with torch.no_grad():
                goal_z = sphere(self.psi(goals))
            z = torch.where(torch.rand(n, 1, device=device) < self.p_goal, goal_z, z)
        return z

    def policy(self, z: Tensor, task: Tensor, explore: bool = False) -> Tensor:
        """Actions ``(..., action_dim)`` in ``[-1, 1]`` for state embeddings ``z`` and tasks ``task``: the actor's
        mean, or with ``explore`` a sample with clipped Gaussian noise."""
        mean = self.actor(z, task)
        if not explore:
            return mean
        noise = (self.actor_std * torch.randn_like(mean)).clamp(-self.actor_clip, self.actor_clip)
        return (mean + noise).clamp(-1, 1)

    def predict(self, z: Tensor, actions: Tensor, task: Tensor, target: bool = False, detach: bool = False) -> Tensor:
        """``T_phi``: successor features of ``psi`` ``(..., task_dim)`` under ``pi_task`` from ``(z, actions)``, the
        mean of the twin heads. ``detach`` lets gradients reach the inputs but not the predictor."""
        net = self.targets["phi_predictor"] if target else self.phi_predictor
        if detach:
            return functional_call(net, {k: v.detach() for k, v in net.named_parameters()}, (z, actions, task)).mean(0)
        return net(z, actions, task).mean(0)

    def value(self, z: Tensor, actions: Tensor, task: Tensor, **kwargs) -> Tensor:
        """Q-values ``(...)`` of ``pi_task`` for the reward ``psi(s) . task``: ``T_phi(z, a, task) . task``."""
        return (self.predict(z, actions, task, **kwargs) * task).sum(-1)

    @torch.no_grad()
    def _psi_batched(self, obs: Tensor, chunk: int = 1024) -> Tensor:
        device = next(self.parameters()).device
        return torch.cat([self.psi(x.to(device)) for x in obs.split(chunk)])

    @torch.no_grad()
    def infer_task(self, obs: Tensor, rewards: Tensor, ridge: float = 1e-6, project: bool = True) -> Tensor:
        """Zero-shot task embedding ``(task_dim,)`` for the reward ``rewards`` ``(N,)`` observed at ``obs`` ``(N,
        ...)``: the least-squares ``z = E[psi psi^T]^-1 E[psi r]``, projected onto the sphere the policies were
        trained on unless ``project=False``."""
        psi = self._psi_batched(obs)
        cov = psi.T @ psi / len(psi)
        z = torch.linalg.solve(cov + ridge * torch.eye(len(cov), device=cov.device), psi.T @ rewards.to(psi) / len(psi))
        return sphere(z) if project else z

    @torch.no_grad()
    def goal_task(self, goals: Tensor, obs: Tensor | None = None, ridge: float = 1e-6) -> Tensor:
        """Task embeddings ``(B, task_dim)`` for reaching goal observations ``(B, ...)``: ``psi(goal)``, whitened by
        ``E[psi psi^T]^-1`` over ``obs`` ``(N, ...)`` when given (the regression of a reward that is one at the goal),
        and projected onto the sphere."""
        z = self._psi_batched(goals)
        if obs is not None:
            psi = self._psi_batched(obs)
            cov = psi.T @ psi / len(psi) + ridge * torch.eye(psi.size(1), device=psi.device)
            z = torch.linalg.solve(cov, z.T).T
        return sphere(z)

    def controller(self, task: Tensor) -> Callable[[Tensor], Tensor]:
        """Closed-loop policy ``observation (B, ...) -> action (B, action_dim)`` for tasks ``(B, task_dim)`` or one
        task ``(task_dim,)`` shared by the batch; it runs the actor's mean on the model's device."""
        device = next(self.parameters()).device

        @torch.no_grad()
        def act(obs: Tensor) -> Tensor:
            z = self.phi(obs.to(device))
            return self.policy(z, task.to(device).expand(len(z), -1)).to(obs.device)

        return act

    @staticmethod
    def orthonormality(x: Tensor) -> Tensor:
        """``mean_{i != j} (x_i . x_j)^2 / 2 - mean_i x_i . x_i`` over a batch ``(B, d)``: pushes ``E[x x^T]`` to
        ``I``."""
        cov = x @ x.T
        off = ~torch.eye(len(x), dtype=torch.bool, device=x.device)
        return 0.5 * cov[off].square().mean() - cov.diagonal().mean()

    def forward(self, pixels: Tensor, actions: Tensor, goals: Tensor | None = None) -> dict[str, Tensor]:
        """Transitions ``pixels`` ``(B, 2, ...)`` (``s`` and ``s'``) and ``actions`` ``(B, 2, action_dim)`` (only
        ``actions[:, 0]``, taken at ``s``, is used), as windows of 2 from ``TrajectoryDataset``. Goal tasks are
        ``psi`` of ``goals`` ``(B, ...)``, by default the batch's next observations shuffled."""
        obs, action, next_obs = pixels[:, 0], actions[:, 0], pixels[:, 1]
        with torch.no_grad():
            z = self.sample_tasks(len(obs), next_obs[torch.randperm(len(obs))] if goals is None else goals)
            next_phi, next_psi = self.phi(next_obs, target=True), self.psi(next_obs, target=True)
            next_action = self.policy(next_phi, z, explore=True)
            phi_target = next_psi + self.discount * self.targets["phi_predictor"](next_phi, next_action, z).mean(0)
            psi_target = next_phi + self.discount * self.targets["psi_predictor"](next_psi, next_action, z).mean(0)
        phi, psi = self.phi(obs), self.psi(obs)
        td_phi = 0.5 * (self.phi_predictor(phi, action, z) - phi_target).square().sum(-1).mean()
        td_psi = 0.5 * (self.psi_predictor(psi, action, z) - psi_target).square().sum(-1).mean()
        ortho = self.orthonormality(phi) + self.orthonormality(psi)

        state = phi.detach()
        actor = -self.value(state, self.policy(state, z, explore=True), z, detach=True).mean()
        loss = td_phi + td_psi + self.lamb * ortho + actor
        return {"loss": loss, "td_phi_loss": td_phi, "td_psi_loss": td_psi, "ortho_loss": ortho, "actor_loss": actor}
