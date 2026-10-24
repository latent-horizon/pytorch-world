import torch
from torch import Tensor


class MPPI:
    """TD-MPC2's planner: model predictive path integral control over returns estimated in latent space.

    Each iteration scores ``num_samples`` action sequences with ``model.estimate_value``: ``num_pi_trajs`` sampled
    from the model's policy prior ``model.pi`` and the rest from a diagonal Gaussian, clamped to ``[-1, 1]``. The
    Gaussian is refit to the ``num_elites`` best, weighted by ``softmax(temperature * value)``, with its std clamped to
    ``[min_std, max_std]``. The returned plan is one elite drawn with those weights.

    Given a ``goal`` latent, MPPI plans for any ``WorldModel`` instead: the value of a sequence is minus
    ``model.cost`` of its rollout from the context, and no trajectories come from a policy prior. ``temperature`` then
    scales costs, so it depends on their range.

    Consecutive calls start from the previous mean shifted by one step; call ``reset`` between episodes.
    """

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        num_samples: int = 512,
        num_elites: int = 64,
        num_pi_trajs: int = 24,
        iterations: int = 6,
        min_std: float = 0.05,
        max_std: float = 2.0,
        temperature: float = 0.5,
    ):
        if not num_pi_trajs < num_samples or not 0 < num_elites <= num_samples:
            raise ValueError("MPPI needs num_pi_trajs < num_samples and 0 < num_elites <= num_samples")
        self.horizon, self.action_dim = horizon, action_dim
        self.num_samples, self.num_elites, self.num_pi_trajs = num_samples, num_elites, num_pi_trajs
        self.iterations, self.min_std, self.max_std, self.temperature = iterations, min_std, max_std, temperature
        self.mean: Tensor | None = None

    def reset(self):
        self.mean = None

    @torch.no_grad()
    def __call__(
        self,
        model,
        z: Tensor,
        goal: Tensor | None = None,
        generator: torch.Generator | None = None,
        explore: bool = False,
    ) -> Tensor:
        """Plan ``(B, horizon, action_dim)`` from the context latents ``z`` ``(B, T0, ...)``.

        Without ``goal``, the plan maximizes TD-MPC2's ``model.estimate_value`` from the last latent. With a goal
        ``(B, ...)``, it minimizes ``model.cost`` of the rollout. ``explore`` adds Gaussian noise with the final std
        to the plan, as the reference does while collecting data.
        """
        B, S, H, A = z.size(0), self.num_samples, self.horizon, self.action_dim
        P = self.num_pi_trajs if goal is None else 0
        if goal is None:
            z = z[:, -1].repeat_interleave(S, dim=0)
        else:
            z, goal = z.repeat_interleave(S, dim=0), goal.repeat_interleave(S, dim=0)
        actions = z.new_empty(B, S, H, A)
        if P:
            zp = z.view(B, S, -1)[:, :P].flatten(0, 1)
            for t in range(H):
                actions[:, :P, t] = model.pi(zp)[0].view(B, P, A)
                if t < H - 1:
                    zp = model.predict(zp, actions[:, :P, t].flatten(0, 1))

        mean = z.new_zeros(B, H, A)
        if self.mean is not None and self.mean.shape == mean.shape:
            mean[:, :-1] = self.mean[:, 1:]
        std = torch.full_like(mean, self.max_std)
        batch = torch.arange(B, device=z.device)[:, None]
        for _ in range(self.iterations):
            noise = torch.randn(B, S - P, H, A, device=z.device, dtype=z.dtype, generator=generator)
            actions[:, P:] = (mean[:, None] + std[:, None] * noise).clamp(-1, 1)
            if goal is None:
                value = model.estimate_value(z, actions.flatten(0, 1))
            else:
                value = -model.cost(model.rollout(z, actions.flatten(0, 1)), goal)
            value = value.nan_to_num(0).view(B, S)
            elite_value, idx = value.topk(self.num_elites, dim=1)
            elites = actions[batch, idx]
            score = (self.temperature * (elite_value - elite_value.max(1, keepdim=True).values)).exp()
            score = (score / score.sum(1, keepdim=True))[..., None, None]
            mean = (score * elites).sum(1)
            std = (score * (elites - mean[:, None]).square()).sum(1).sqrt().clamp(self.min_std, self.max_std)
        self.mean = mean

        choice = torch.multinomial(score[..., 0, 0], 1, generator=generator)
        plan = elites[batch, choice][:, 0]
        if explore:
            plan = plan + std * torch.randn(plan.shape, device=z.device, dtype=z.dtype, generator=generator)
        return plan.clamp(-1, 1)
