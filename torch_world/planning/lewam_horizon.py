"""Single-view MSE-dynamics counterpart of upstream LeWAM gradient planning.

Reference: MinghaoFu/lewam f57659f, lewam/eval/planners.py (MIT).
Supports sampled-goal checkpoints, strided history and optimization of all
proposals. CEM, flow-state dynamics and multi-view models are outside this path.
"""

import torch
from torch.nn import functional as F


class LeWAMHorizonPlanner:
    """Imagine policy rollouts, optimize normalized actions with Adam, select best.

    ``steps_to_goal`` counts dynamics steps, not raw controls. The history must
    use the same spacing as ``model.frameskip``. Execute one frameskip block,
    then replan with actual observations. Model weights and gradients are never
    updated. Returned actions are bounded physical controls; internal imagined
    proposals use the upstream normalized-action convention.
    """

    def __init__(
        self,
        *,
        num_samples=32,
        horizon=10,
        refinement_steps=5,
        lr=0.05,
        proximity=0.01,
        grad_clip=10.0,
        action_clip=3.0,
    ):
        if (
            min(num_samples, horizon) < 1
            or refinement_steps < 0
            or lr <= 0
            or min(proximity, grad_clip, action_clip) < 0
        ):
            raise ValueError("Invalid horizon planning budget")
        self.num_samples, self.horizon = num_samples, horizon
        self.refinement_steps, self.lr, self.proximity = refinement_steps, lr, proximity
        self.grad_clip, self.action_clip = grad_clip, action_clip
        self.last_diagnostics = {}

    @staticmethod
    def _next(model, history, pad, normalized):
        clean = F.pad(normalized, (0, 0, 0, model.chunk_size - model.frameskip))
        future = model.predict_future(history, model.denormalize_actions(clean), history_pad=pad)[:, :1]
        return torch.cat((history[:, 1:], future), 1), torch.cat((pad[:, 1:], torch.zeros_like(pad[:, :1])), 1)

    def __call__(self, model, z, goal, *, steps_to_goal, history_pad=None, generator=None):
        if model.training or getattr(model, "goal_type", None) != "sampled":
            raise ValueError("Horizon planning requires an eval-mode sampled-goal checkpoint")
        if model.num_states != 1 or model.state_dim:
            raise ValueError("Horizon planner supports one visual MSE state per dynamics step")
        b, n = len(z), self.num_samples
        left = torch.as_tensor(steps_to_goal, device=z.device, dtype=z.dtype)
        if left.shape != (b,) or not torch.isfinite(left).all() or (left <= 0).any():
            raise ValueError("steps_to_goal must be finite positive (B,) dynamics-step counts")
        index = left.round().clamp(1, self.horizon).long() - 1
        length = int(index.max()) + 1
        history = z.detach().repeat_interleave(n, 0)
        target = goal.detach().repeat_interleave(n, 0)
        pad = torch.zeros(z.shape[:2], dtype=torch.bool, device=z.device) if history_pad is None else history_pad
        pad = pad.repeat_interleave(n, 0)
        index, left = index.repeat_interleave(n), left.repeat_interleave(n)
        rows = torch.arange(b * n, device=z.device)

        def rollout(plan):
            h, p, states = history, pad, []
            for step in range(length):
                h, p = self._next(model, h, p, plan[:, step])
                states.append(h[:, -1])
            at_goal = torch.stack(states, 1)[rows, index]
            return (at_goal - target).square().mean(-1)

        with torch.no_grad():
            h, p, proposals = history, pad, []
            for step in range(length):
                actions = model.sample_actions(
                    h,
                    target,
                    remaining=(left - step).clamp(1, self.horizon) / self.horizon,
                    history_pad=p,
                    generator=generator,
                    clamp=False,
                )
                block = model.normalize_actions(actions[:, : model.frameskip])
                proposals.append(block)
                h, p = self._next(model, h, p, block)
            anchor = torch.stack(proposals, 1)
            best, best_cost = anchor.clone(), rollout(anchor)
            warm_cost = best_cost.view(b, n).min(1).values
        with torch.enable_grad():
            plan = anchor.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([plan], lr=self.lr)
            for step in range(self.refinement_steps + 1):
                cost = rollout(plan)
                with torch.no_grad():
                    better = cost < best_cost
                    best[better], best_cost[better] = plan[better], cost[better]
                if step == self.refinement_steps:
                    break
                loss = cost.sum() + self.proximity * (plan - anchor).square().sum()
                optimizer.zero_grad(set_to_none=True)
                plan.grad = torch.autograd.grad(loss, plan)[0]
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_([plan], self.grad_clip)
                optimizer.step()
                if self.action_clip:
                    with torch.no_grad():
                        plan.clamp_(-self.action_clip, self.action_clip)
        choices = best_cost.view(b, n).argmin(1)
        selected = best.view(b, n, length * model.frameskip, model.action_dim)[
            torch.arange(b, device=z.device), choices
        ]
        self.last_diagnostics = {
            "warm_cost": warm_cost.detach(),
            "best_cost": best_cost.view(b, n).min(1).values.detach(),
        }
        return model.denormalize_actions(selected, clamp=True).detach()
