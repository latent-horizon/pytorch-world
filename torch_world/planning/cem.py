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
        mean = z.new_zeros(B, self.horizon, self.action_dim)
        std = torch.full_like(mean, self.std)
        z, goal, cond = _repeat(z, S), _repeat(goal, S), {k: _repeat(v, S) for k, v in cond.items()}
        past = _repeat(past_actions, S) if past_actions is not None else mean.new_zeros(B * S, 0, self.action_dim)
        for _ in range(self.iterations):
            noise = torch.randn(
                B, S, self.horizon, self.action_dim, device=z.device, dtype=z.dtype, generator=generator
            )
            samples = mean[:, None] + std[:, None] * noise
            if self.keep_mean:
                samples[:, 0] = mean
            if self.bounds is not None:
                samples = samples.clamp(*self.bounds)
            rollout = model.rollout(z, torch.cat([past, samples.flatten(0, 1)], dim=1), **cond)
            if self.objective == "path":
                cost = torch.stack([model.cost(rollout[:, : t + 1], goal) for t in range(T0, rollout.size(1))]).mean(0)
            else:
                cost = model.cost(rollout, goal)
            cost = cost.view(B, S)
            elites = samples[torch.arange(B)[:, None], cost.topk(self.num_elites, largest=False).indices]
            mean, std = elites.mean(1), elites.std(1)
        return mean


def _repeat(x, n: int):
    return x.repeat_interleave(n, dim=0) if isinstance(x, Tensor) else x
