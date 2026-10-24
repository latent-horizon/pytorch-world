import torch
from torch import Tensor

from torch_world.models.base import WorldModel
from torch_world.planning.cem import _repeat


class GRASP:
    """Gradient RelAxed Stochastic Planner: gradient descent on actions and on free "virtual" intermediate latents.

    Instead of backpropagating through a long serial rollout, the latents ``s_1 .. s_{H-1}`` between the current
    latent ``s_0`` and the goal ``g`` (``s_H = g``) become optimization variables, initialized on the line from
    ``s_0`` to ``g``. Every step is then predicted in parallel from its own (stop-gradient) start latent:

        loss = sum_t |F(sg(s_t), a_t) - s_{t+1}|^2 + goal_weight * sum_t cost(F(sg(s_t), a_t), g)

    Actions get gradients through the model, latents only from the dynamics term, never through the model's state
    Jacobian. Both take an Adam step (``lr_states``, ``lr_actions``), then Gaussian noise ``noise`` is added to the
    latents to explore. Every ``sync_every`` steps the actions are refined by ``sync_steps`` Adam steps
    (``sync_lr``; ``sync_every=0`` disables it) on the true objective ``model.cost`` of a full rollout from the context, which keeps the relaxed
    plan dynamically feasible, and the joint optimizer restarts. The first step is predicted from the full context
    ``z`` (and ``past_actions``); the others from their single virtual latent.

    Psenka et al., "Parallel Stochastic Gradient-Based Planning for World Models", 2026.
    Paper: https://arxiv.org/abs/2602.00475
    Code: https://github.com/michael-psenka/grasp (LeWM adapter: https://github.com/michael-psenka/le-wm-grasp)
    """

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        iterations: int = 750,
        lr_states: float = 0.1,
        lr_actions: float = 0.005,
        noise: float = 0.005,
        goal_weight: float = 1.0,
        sync_every: int = 50,
        sync_steps: int = 10,
        sync_lr: float = 0.05,
        bounds: tuple[float, float] | None = None,
    ):
        self.horizon, self.action_dim, self.iterations = horizon, action_dim, iterations
        self.lr_states, self.lr_actions, self.noise, self.goal_weight = lr_states, lr_actions, noise, goal_weight
        self.sync_every, self.sync_steps, self.sync_lr = sync_every, sync_steps, sync_lr
        self.bounds = bounds

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
        batch dim are forwarded to ``model.rollout``. Runs with gradients enabled even under ``torch.no_grad``; the
        model's parameters get none.
        """
        B, H = z.size(0), self.horizon
        z, goal = z.detach(), goal.detach()
        past = z.new_zeros(B, 0, self.action_dim) if past_actions is None else past_actions
        frac = torch.arange(1, H, device=z.device, dtype=z.dtype).view(1, -1, *[1] * (goal.dim() - 1)) / H
        states = (z[:, -1:] + frac * (goal - z[:, -1])[:, None]).requires_grad_()
        actions = z.new_zeros(B, H, self.action_dim, requires_grad=True)
        cond_steps = {k: _repeat(v, H - 1) for k, v in cond.items()}
        goal_steps = goal.repeat_interleave(H, dim=0)

        def joint_optimizer():
            return torch.optim.Adam(
                [{"params": [states], "lr": self.lr_states}, {"params": [actions], "lr": self.lr_actions}]
            )

        opt = joint_optimizer()
        with torch.enable_grad():
            for k in range(self.iterations):
                pred = model.rollout(z, torch.cat([past, actions[:, :1]], dim=1), **cond)[:, -1:]
                if H > 1:
                    s, a = states.detach().flatten(0, 1)[:, None], actions[:, 1:].flatten(0, 1)[:, None]
                    rest = model.rollout(s, a, **cond_steps)[:, -1].unflatten(0, (B, H - 1))
                    pred = torch.cat([pred, rest], dim=1)
                dynamics = (pred - torch.cat([states, goal[:, None]], dim=1)).square().flatten(2).mean(2).sum(1)
                to_goal = model.cost(pred.flatten(0, 1)[:, None], goal_steps).view(B, H).sum(1)
                loss = (dynamics + self.goal_weight * to_goal).sum()
                states.grad, actions.grad = torch.autograd.grad(loss, [states, actions])
                opt.step()
                with torch.no_grad():
                    states.add_(
                        self.noise * torch.randn(states.shape, device=z.device, dtype=z.dtype, generator=generator)
                    )
                    if self.bounds is not None:
                        actions.clamp_(*self.bounds)
                if self.sync_every and (k + 1) % self.sync_every == 0:
                    self._sync(model, z, goal, past, actions, cond)
                    opt = joint_optimizer()
        return actions.detach()

    def _sync(self, model, z, goal, past, actions, cond):
        """Adam on the actions alone against the cost of a full rollout from the context."""
        opt = torch.optim.Adam([actions], lr=self.sync_lr)
        for _ in range(self.sync_steps):
            cost = model.cost(model.rollout(z, torch.cat([past, actions], dim=1), **cond), goal).sum()
            (actions.grad,) = torch.autograd.grad(cost, [actions])
            opt.step()
            if self.bounds is not None:
                with torch.no_grad():
                    actions.clamp_(*self.bounds)
