import copy
import math

import torch
from torch import Tensor, nn
from torch.func import functional_call

from torch_world.models.base import WorldModel
from torch_world.nn.transformer import MLP
from torch_world.safety.margin import FailureMargin, flat
from torch_world.utils.ema import ema_update


class SafetyValue(nn.Module):
    """Safety value function of a latent world model, learned by discounted Hamilton-Jacobi reachability.

    The value ``V(z)`` is the best worst-case safety margin reachable from latent ``z``: the lowest ``l`` the system
    will ever see when the safety policy acts. Its sign splits the latent space into states from which failure can be
    avoided forever (``V > 0``) and states from which it cannot. With continuous actions it is learned as in the paper
    with DDPG: a critic ``Q(z, a)`` regresses the discounted avoid backup on imagined transitions ``(z, a, z')`` of
    the frozen world model, with target networks ``Q'``, ``pi'`` updated by Polyak averaging, and an actor ``pi(z)``
    (the safety policy) maximizes the critic:

    ``Q(z, a) <- (1 - gamma) l(z) + gamma min(l(z), Q'(z', pi'(z')))``,   ``V(z) = Q(z, pi(z))``

    ``critic_loss = (Q(z, a) - target)^2``, ``actor_loss = -Q(z, pi(z))`` (the critic held fixed).

    Nakamura, Peters and Bajcsy, "Generalizing Safety Beyond Collision-Avoidance via Latent-Space Reachability Analysis", 2025.
    Paper: https://arxiv.org/abs/2502.00935
    Code: https://github.com/CMU-IntentLab/latent-safety

    Args:
        latent_shape: shape of one latent of the world model; latents are flattened.
        action_dim: dimension of one world-model action.
        hidden: hidden widths of the actor and the critic (the paper's DDPG uses four layers of 512).
        gamma: discount; the paper uses 0.9999. ``gamma < 1`` makes the backup a contraction.
        tau: Polyak rate of the target networks (the reference code's 0.005).
        bounds: action range of the safety policy, reached through a ``tanh``.
    """

    def __init__(
        self,
        latent_shape: int | tuple[int, ...],
        action_dim: int,
        hidden: tuple[int, ...] = (512, 512, 512, 512),
        gamma: float = 0.9999,
        tau: float = 0.005,
        bounds: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.shape = (latent_shape,) if isinstance(latent_shape, int) else tuple(latent_shape)
        dim = math.prod(self.shape)
        self.actor = MLP([dim, *hidden, action_dim], act=nn.ReLU)
        self.critic = MLP([dim + action_dim, *hidden, 1], act=nn.ReLU)
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.action_dim, self.gamma, self.tau, self.bounds = action_dim, gamma, tau, bounds

    def action(self, z: Tensor, target: bool = False) -> Tensor:
        """Safety policy ``pi(z)``: latents ``(N, *latent_shape)`` to actions ``(N, action_dim)`` inside ``bounds``."""
        low, high = self.bounds
        u = (self.actor_target if target else self.actor)(flat(z, self.shape)).tanh()
        return low + (u + 1) * (high - low) / 2

    def q(self, z: Tensor, action: Tensor, target: bool = False, params: dict | None = None) -> Tensor:
        """Critic ``Q(z, a)`` ``(N,)``; ``params`` evaluates it with other parameters (for example detached ones)."""
        x = torch.cat([flat(z, self.shape), action], -1)
        critic = self.critic_target if target else self.critic
        return (critic(x) if params is None else functional_call(critic, params, (x,)))[..., 0]

    def value(self, z: Tensor) -> Tensor:
        """Safety value ``V(z) = Q(z, pi(z))`` ``(N,)``; positive where failure can be avoided."""
        return self.q(z, self.action(z))

    def target(self, margin: Tensor, z_next: Tensor, gamma: float | None = None) -> Tensor:
        """Discounted avoid backup ``(1 - gamma) l(z) + gamma min(l(z), Q'(z', pi'(z')))`` ``(N,)``."""
        gamma = self.gamma if gamma is None else gamma
        with torch.no_grad():
            future = self.q(z_next, self.action(z_next, target=True), target=True)
            return (1 - gamma) * margin + gamma * torch.minimum(margin, future)

    def forward(
        self, z: Tensor, actions: Tensor, z_next: Tensor, margin: Tensor, gamma: float | None = None
    ) -> dict[str, Tensor]:
        """Losses on transitions: latents ``z`` and ``z_next`` ``(N, *latent_shape)``, ``actions`` ``(N, action_dim)``
        and margins ``l(z)`` ``(N,)``. ``gamma=0`` regresses ``Q`` on ``l`` alone (the reference's warm-up)."""
        critic_loss = (self.q(z, actions) - self.target(margin, z_next, gamma)).square().mean()
        frozen = {k: v.detach() for k, v in self.critic.named_parameters()}
        actor_loss = -self.q(z, self.action(z), params=frozen).mean()
        return {"loss": critic_loss + actor_loss, "critic_loss": critic_loss, "actor_loss": actor_loss}

    def update_target(self):
        """Polyak update of both target networks with rate ``tau``."""
        ema_update(self.actor_target, self.actor, 1 - self.tau)
        ema_update(self.critic_target, self.critic, 1 - self.tau)


@torch.no_grad()
def imagine_step(model: WorldModel, z: Tensor, actions: Tensor) -> Tensor:
    """One imagined step of the world model from single latents ``(N, *latent_shape)``: ``(N, *latent_shape)``."""
    return model.rollout(z[:, None], actions[:, None])[:, -1]


def fit_safety_value(
    value: SafetyValue,
    model: WorldModel,
    margin: FailureMargin,
    starts: Tensor,
    iterations: int,
    warmup: int = 0,
    horizon: int = 25,
    envs: int = 8,
    updates: int = 1,
    batch_size: int = 512,
    buffer_size: int = 40_000,
    actor_lr: float = 1e-4,
    critic_lr: float = 1e-3,
    noise: float = 0.1,
    generator: torch.Generator | None = None,
) -> dict[str, list[float]]:
    """Train ``value`` by reachability in imagination; the world model and the margin stay frozen.

    ``envs`` imagined episodes run in parallel. Each starts from a latent drawn from ``starts`` ``(N, *latent_shape)``
    (encoded dataset frames) and runs ``horizon`` steps of ``model`` before it is reset, to limit the drift away from
    the data (the paper's 25). Each iteration steps every episode once, stores the transitions with ``l(z)`` in a
    replay buffer of ``buffer_size`` and makes ``updates`` AdamW steps on batches of ``batch_size``. Exploration
    follows the reference code: the actor's action plus Gaussian ``noise``, replaced by a uniform random action
    whenever the critic judges that one safe (``Q >= 0``). The first ``warmup`` iterations act at random and use
    ``gamma = 0``, fitting ``Q`` to ``l`` before the backup starts. Returns the ``critic`` and ``actor`` losses.
    """
    device = starts.device
    model.eval().requires_grad_(False)
    margin.eval().requires_grad_(False)
    opt = torch.optim.AdamW(
        [{"params": value.actor.parameters(), "lr": actor_lr}, {"params": value.critic.parameters(), "lr": critic_lr}]
    )
    low, high = value.bounds
    shape, A = starts.shape[1:], value.action_dim
    buffer = {
        "z": starts.new_zeros(buffer_size, *shape),
        "actions": starts.new_zeros(buffer_size, A),
        "z_next": starts.new_zeros(buffer_size, *shape),
        "margin": starts.new_zeros(buffer_size),
    }
    size, head = 0, 0

    def draw(n: int) -> Tensor:
        return starts[torch.randint(len(starts), (n,), generator=generator).to(device)]

    def uniform(n: int) -> Tensor:
        return (low + (high - low) * torch.rand(n, A, generator=generator)).to(device)

    z, t = draw(envs), 0
    log: dict[str, list[float]] = {"critic": [], "actor": []}
    for i in range(warmup + iterations):
        warm = i < warmup
        with torch.no_grad():
            if warm:
                a = uniform(envs)
            else:
                a = value.action(z) + noise * torch.randn(envs, A, generator=generator).to(device)
                a = a.clamp(low, high)
                random = uniform(envs)
                a = torch.where((value.q(z, random) >= 0)[:, None], random, a)
            z_next = imagine_step(model, z, a)
            idx = (head + torch.arange(envs, device=device)) % buffer_size
            for k, v in zip(buffer, (z, a, z_next, margin.margin(z))):
                buffer[k][idx] = v
            head, size = (head + envs) % buffer_size, min(size + envs, buffer_size)
            t += 1
            z, t = (draw(envs), 0) if t == horizon else (z_next, t)
        for _ in range(updates):
            b = torch.randint(size, (batch_size,), generator=generator).to(device)
            out = value(*(buffer[k][b] for k in buffer), gamma=0.0 if warm else None)
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            value.update_target()
            log["critic"].append(out["critic_loss"].item())
            log["actor"].append(out["actor_loss"].item())
    value.eval()
    return log
