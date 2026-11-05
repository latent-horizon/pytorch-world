import torch
from torch import Tensor

from torch_world.models.base import WorldModel


class CEM:
    """Cross-entropy method over action sequences in the latent space of a ``WorldModel``.

    Each iteration samples ``num_samples`` sequences from a diagonal Gaussian, rolls them out with the model and
    refits the Gaussian to the ``num_elites`` cheapest under ``model.cost``. The ``"final"`` objective scores the last
    imagined latent; ``"path"`` averages the cost of every imagined latent, which rewards getting close early. Under
    MPC, where only the first action runs, ``"path"`` keeps the planner from postponing its moves. With ``keep_mean``
    the first sample of every iteration is the current mean, as in stable-worldmodel's CEM, so a good mean is never
    lost to an unlucky draw.

    Costs are ranked in fp32 and the sampling distribution is kept in fp32 even when the caller runs under
    ``torch.autocast``: bf16 keeps about three significant digits, too few to order close candidate costs.
    """

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        num_samples: int = 300,
        num_elites: int = 30,
        iterations: int = 10,
        std: float = 1.0,
        bounds: tuple[float, float] | None = None,
        objective: str = "final",
        keep_mean: bool = False,
    ):
        if objective not in ("final", "path"):
            raise ValueError(f"objective must be 'final' or 'path', not {objective!r}")
        self.horizon, self.action_dim = horizon, action_dim
        self.num_samples, self.num_elites, self.iterations = num_samples, num_elites, iterations
        self.std = std
        self.bounds = bounds
        self.objective = objective
        self.keep_mean = keep_mean

    @torch.no_grad()
    def __call__(
        self,
        model: WorldModel,
        z: Tensor,
        goal: Tensor,
        past_actions: Tensor | None = None,
        generator: torch.Generator | None = None,
        **cond,
    ) -> Tensor:
        """Plan ``(B, horizon, action_dim)`` from context latents ``z`` ``(B, T0, ...)`` toward ``goal`` ``(B, ...)``.

        ``past_actions`` ``(B, T0 - 1, action_dim)`` are the actions between context frames; ``cond`` tensors with a
        batch dim are forwarded to ``model.rollout``.
        """
        B, S, T0 = z.size(0), self.num_samples, z.size(1)
        mean = z.new_zeros(B, self.horizon, self.action_dim, dtype=torch.float32)
        std = torch.full_like(mean, self.std)
        z, goal, cond = _repeat(z, S), _repeat(goal, S), {k: _repeat(v, S) for k, v in cond.items()}
        past = _repeat(past_actions, S) if past_actions is not None else z.new_zeros(B * S, 0, self.action_dim)
        for _ in range(self.iterations):
            noise = torch.randn(B, S, self.horizon, self.action_dim, device=z.device, generator=generator)
            samples = mean[:, None] + std[:, None] * noise
            if self.keep_mean:
                samples[:, 0] = mean
            if self.bounds is not None:
                samples = samples.clamp(*self.bounds)
            rollout = model.rollout(z, torch.cat([past, samples.flatten(0, 1).to(past.dtype)], dim=1), **cond)
            if self.objective == "path":
                cost = torch.stack([fp32_cost(model, rollout[:, : t + 1], goal) for t in range(T0, rollout.size(1))])
                cost = cost.mean(0)
            else:
                cost = fp32_cost(model, rollout, goal)
            cost = cost.view(B, S)
            elites = samples[torch.arange(B)[:, None], cost.topk(self.num_elites, largest=False).indices]
            mean, std = elites.mean(1), elites.std(1)
        return mean.to(z.dtype)


def fp32_cost(model, rollout: Tensor, goal: Tensor) -> Tensor:
    """``model.cost(rollout, goal)`` in fp32, for ranking candidates.

    Under ``torch.autocast`` the cost runs with autocast disabled on fp32 copies of its inputs, so a cost network
    with fp32 weights scores in full precision; otherwise the inputs keep their dtype and only the result is upcast.
    """
    device = rollout.device.type
    if torch.is_autocast_enabled(device):
        with torch.autocast(device, enabled=False):
            return model.cost(rollout.float(), goal.float()).float()
    return model.cost(rollout, goal).float()


def _repeat(x, n: int):
    return x.repeat_interleave(n, dim=0) if isinstance(x, Tensor) else x
