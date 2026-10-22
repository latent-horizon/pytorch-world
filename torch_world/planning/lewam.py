"""Learned world-action-model proposals, optionally ranked/refined by the model's own dynamics."""

import math

import torch


class LeWAMPlanner:
    """Sample action chunks, optionally refine and rank them toward a latent goal.

    Works with any ``WorldActionModel`` (LeWAM, DreamZero); ``WAMPlanner`` is the same class. Extra keyword
    conditions, such as DreamZero's ``past_actions`` and ``states``, are repeated per sample and forwarded to
    ``sample_actions`` and ``predict_future``; ``goal`` is forwarded to both too.

    ``num_samples=1, refinement_steps=0`` evaluates the direct learned policy.
    More samples rank proposals using future latent goal distance. Optional
    gradient refinement penalizes movement away from each learned proposal.
    Only actions receive gradients; model parameters and context are untouched.
    This planner uses the model's action normalization and physical bounds.
    The caller executes the first action and replans from a new observation.
    ``flow_steps`` optionally overrides the model's integration budget.
    ``noise_scale=0`` starts at the Gaussian centre for deterministic diagnostics;
    this is not the mean of the learned action distribution. Defaults preserve
    the model's ordinary Gaussian sampling and integration budget.
    """

    def __init__(self, *, num_samples=1, refinement_steps=0, lr=0.05, proximity=0.1, noise_scale=1.0, flow_steps=None):
        if num_samples < 1 or refinement_steps < 0 or lr <= 0 or proximity < 0:
            raise ValueError("Invalid LeWAM planning budget")
        if not math.isfinite(noise_scale) or noise_scale < 0:
            raise ValueError("noise_scale must be finite and nonnegative")
        if flow_steps is not None and (not isinstance(flow_steps, int) or flow_steps < 1):
            raise ValueError("flow_steps must be a positive integer")
        self.num_samples, self.refinement_steps = num_samples, refinement_steps
        self.lr, self.proximity = lr, proximity
        self.noise_scale, self.flow_steps = noise_scale, flow_steps

    def __call__(self, model, z, goal, past_actions=None, generator=None, *, history_pad=None, remaining=None, **cond):
        b, n = z.size(0), self.num_samples
        if past_actions is not None:
            cond["past_actions"] = past_actions
        cond = {k: v if v is None else v.detach().repeat_interleave(n, 0) for k, v in cond.items()}
        context = z.detach().repeat_interleave(n, 0)
        targets = goal.detach().repeat_interleave(n, 0)
        padding = None if history_pad is None else history_pad.repeat_interleave(n, 0)
        horizon = None if remaining is None else remaining.detach().repeat_interleave(n, 0)
        with torch.no_grad():
            noise = None
            if self.noise_scale != 1:
                shape = (b * n, model.chunk_size, model.action_dim)
                noise = (
                    z.new_zeros(shape)
                    if self.noise_scale == 0
                    else torch.randn(shape, device=z.device, dtype=z.dtype, generator=generator) * self.noise_scale
                )
            proposals = model.sample_actions(
                context,
                targets,
                generator=generator,
                noise=noise,
                steps=self.flow_steps,
                history_pad=padding,
                remaining=horizon,
                **cond,
            )
        if n == 1 and not self.refinement_steps:
            return proposals
        anchor = model.normalize_actions(proposals).detach()
        actions = anchor.clone()

        def score(normalized):
            future = model.predict_future(
                context, model.denormalize_actions(normalized), targets, history_pad=padding, **cond
            )
            return model.cost(future, targets) + self.proximity * (normalized - anchor).square().mean((1, 2))

        with torch.no_grad():
            best, best_cost = actions.clone(), score(actions)
        for _ in range(self.refinement_steps):
            with torch.enable_grad():
                actions = actions.detach().requires_grad_(True)
                gradient = torch.autograd.grad(score(actions).sum(), actions)[0]
            with torch.no_grad():
                actions = model.normalize_actions(model.denormalize_actions(actions - self.lr * gradient, clamp=True))
                cost = score(actions)
                better = cost < best_cost
                best[better], best_cost[better] = actions[better], cost[better]
        index = best_cost.reshape(b, n).argmin(1)
        chunks = model.denormalize_actions(best, clamp=True).reshape(b, n, model.chunk_size, model.action_dim)
        return chunks[torch.arange(b, device=z.device), index].detach()
