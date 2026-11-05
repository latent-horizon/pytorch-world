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
    latents to explore. Every ``sync_every`` steps (``0`` disables it) the actions are refined against the true
    objective, ``model.cost`` of a full rollout from the context, which keeps the relaxed plan dynamically feasible,
    and the joint optimizer restarts. The first step is predicted from the full context ``z`` (and ``past_actions``);
    the others from their single virtual latent.

    The sync is one of the authors' two variants. ``sync="gd"`` takes ``sync_steps`` Adam steps (``sync_lr``) on the
    actions. ``sync="cem"`` runs ``sync_steps`` CEM iterations of ``sync_samples`` samples and ``sync_elites`` elites
    around the current actions; the first iteration's variance per step is ``sync_var_scale`` times the gap between
    the rollout and the virtual latents (where the relaxation broke down, it explores more) plus ``sync_var_min``.
    The authors' LeWM Push-T result (46% against CEM's 22% at horizon 15) uses ``sync="cem"`` with ``sync_steps=5``.

    With ``warm_start=k``, a call starts its actions from the previous plan shifted by ``k`` steps, the ones executed
    in between, padded with zeros, as stable-worldmodel's policy does when it replans before the plan runs out. Under
    ``RecedingHorizon`` this is its ``execute``. Call ``reset`` between episodes.

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
        sync: str = "gd",
        sync_samples: int = 64,
        sync_elites: int = 8,
        sync_var_scale: float = 1.0,
        sync_var_min: float = 0.01,
        warm_start: int = 0,
    ):
        if sync not in ("gd", "cem"):
            raise ValueError(f"sync must be 'gd' or 'cem', not {sync!r}")
        if sync == "cem" and not 1 < sync_elites <= sync_samples:
            raise ValueError("sync='cem' needs 1 < sync_elites <= sync_samples")
        self.horizon, self.action_dim, self.iterations = horizon, action_dim, iterations
        self.lr_states, self.lr_actions, self.noise, self.goal_weight = lr_states, lr_actions, noise, goal_weight
        self.sync_every, self.sync_steps, self.sync_lr = sync_every, sync_steps, sync_lr
        self.sync, self.sync_samples, self.sync_elites = sync, sync_samples, sync_elites
        self.sync_var_scale, self.sync_var_min = sync_var_scale, sync_var_min
        self.bounds, self.warm_start = bounds, warm_start
        self.plan: Tensor | None = None

    def reset(self):
        self.plan = None

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
        actions = z.new_zeros(B, H, self.action_dim)
        if 0 < self.warm_start < H and self.plan is not None and self.plan.shape == actions.shape:
            actions[:, : H - self.warm_start] = self.plan[:, self.warm_start :]
        actions.requires_grad_()
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
                    if self.sync == "cem":
                        self._cem_sync(model, z, goal, past, actions, states, cond, generator)
                    else:
                        self._sync(model, z, goal, past, actions, cond)
                    opt = joint_optimizer()
        self.plan = actions.detach()
        return self.plan

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

    @torch.no_grad()
    def _cem_sync(self, model, z, goal, past, actions, states, cond, generator):
        """CEM around the actions, with a first variance per step from the gap between rollout and virtual latents."""
        B, H, A, N, T0 = z.size(0), self.horizon, self.action_dim, self.sync_samples, z.size(1)
        mean = actions.detach().clone()
        rollout = model.rollout(z, torch.cat([past, mean], dim=1), **cond)
        if H > 1:
            gap = (rollout[:, T0 : T0 + H - 1] - states).square().flatten(2).mean(2)
            gap = torch.cat([gap.mean(1, keepdim=True), gap], dim=1)
        else:
            gap = (rollout[:, -1] - goal).square().flatten(1).mean(1, keepdim=True)
        var = (self.sync_var_scale * gap + self.sync_var_min)[..., None]
        zs, goals, pasts = _repeat(z, N), _repeat(goal, N), _repeat(past, N)
        conds = {k: _repeat(v, N) for k, v in cond.items()}
        batch = torch.arange(B, device=z.device)[:, None]
        for _ in range(self.sync_steps):
            noise = torch.randn(B, N, H, A, device=z.device, dtype=z.dtype, generator=generator)
            samples = mean[:, None] + noise * var[:, None].sqrt()
            if self.bounds is not None:
                samples = samples.clamp(*self.bounds)
            cost = model.cost(model.rollout(zs, torch.cat([pasts, samples.flatten(0, 1)], dim=1), **conds), goals)
            elites = samples[batch, cost.view(B, N).topk(self.sync_elites, largest=False).indices]
            mean = elites.mean(1)
            var = elites.var(1).mean(-1, keepdim=True) + self.sync_var_min
        actions.copy_(mean)
