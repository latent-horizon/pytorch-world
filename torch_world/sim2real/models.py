"""Model-specific training and planning adapters; shared deployment policy."""

import time
from collections import deque

import torch
import torch.nn.functional as F
from torch import nn

from torch_world.benchmarks.experiment import observation_encoder
from torch_world.models import DINOWM, LeWAM, LeWM, RoboJEPA
from torch_world.nn.encoders import PatchTokens, PixelPatches
from torch_world.planning import CEM, LeWAMPlanner


class ModelAdapter(nn.Module):
    """Keep model-specific layouts here, rather than in the experiment notebook."""

    def __init__(self, model, recipe):
        super().__init__()
        self.model, self.recipe = model, recipe
        self.kind, self.history = recipe["kind"], recipe["history"]
        self.training_window = (
            self.history + (recipe["chunk_size"] if self.kind == "lewam" else recipe.get("rollout_steps", 1)) - 1
        )
        self.grid = tuple(recipe["grid"]) if recipe.get("grid") else None

    @property
    def device(self):
        return next(self.model.parameters()).device

    def proprio(self, state):
        commanded = self.recipe.get("command_state", False)
        if state.size(-1) not in (6, 10) or (commanded and state.size(-1) != 10):
            raise ValueError("Expected measured state (6) or measured + commanded state (10)")
        state = state if commanded else state[..., :6]
        scale = state.new_tensor([torch.pi] * 5 + [1.0] + ([0.5, 0.5, 0.5, 1.0] if commanded else []))
        return state / scale

    def observation_state(self, observation):
        return observation.state(self.recipe.get("command_state", False))

    def encode(self, pixels, states):
        if self.kind in ("lewm", "lewam"):
            return self.model.encode(pixels, self.proprio(states) if self.recipe.get("proprioception") else None)
        if self.kind == "dinowm":
            return self.model.encode(pixels, self.proprio(states))
        return self.model.encode(pixels[:, :, None], self.grid)

    def loss(self, pixels, actions, states, **targets):
        if self.kind == "lewam":
            if targets.get("goal_states") is not None:
                targets["goal_states"] = self.proprio(targets["goal_states"])
            return self.model(
                pixels, actions, self.proprio(states) if self.recipe.get("proprioception") else None, **targets
            )
        if self.kind == "lewm":
            return self.model(pixels, actions, self.proprio(states) if self.recipe.get("proprioception") else None)
        if self.kind == "dinowm":
            return self.model(pixels, actions, self.proprio(states))
        return self.model(self.encode(pixels, states), actions, self.proprio(states))

    def conditions(self, states):
        return {"states": self.proprio(states)} if self.kind == "robojepa" else {}

    def loss_features(self, features, actions, states):
        """DINO-WM loss with cached frozen visual features; proprio stays trainable."""
        if self.kind != "dinowm":
            raise ValueError("Feature caching currently supports DINO-WM only")
        proprio = self.model._tile(self.model.proprio_encoder(self.proprio(states)))
        z = torch.cat((features, proprio), dim=-1)
        prediction = self.model.predict(z[:, : self.history], actions[:, : self.history])
        return {"loss": F.mse_loss(prediction, z[:, 1 : self.history + 1].detach())}


def make_adapter(
    kind="lewm",
    *,
    size=84,
    history=3,
    smoke=False,
    encoder=None,
    encoder_dim=None,
    grid=None,
    proprioception=False,
    rollout_steps=1,
    chunk_size=8,
    action_scale=(1.0, 1.0, 1.0, 0.15),
    command_state=False,
):
    """Small, configurable experiment baselines, not paper-scale reproductions.

    DINO-WM and RoboJEPA use released DINOv2 patch features by default. The
    latter is explicitly a RoboJEPA/DINOv2 variant, not a V-JEPA checkpoint.
    ``smoke=True`` substitutes raw pixel patches and never downloads weights.
    Custom frozen patch encoders must declare their feature dimension and grid.
    """
    if kind not in ("lewm", "lewam", "dinowm", "robojepa") or history < 1:
        raise ValueError("Choose lewm, lewam, dinowm or robojepa, with positive history")
    if command_state and (kind != "lewam" or not proprioception):
        raise ValueError("Commanded state requires LeWAM with proprioception")
    if rollout_steps < 1 or (kind not in ("lewm", "lewam") and (proprioception or rollout_steps != 1)):
        raise ValueError("State inputs and configurable rollout training currently apply to LeWM/LeWAM only")
    recipe = {
        "kind": kind,
        "size": size,
        "history": history,
        "smoke": smoke,
        "encoder": "custom"
        if encoder is not None
        else "cnn"
        if kind in ("lewm", "lewam")
        else "pixels"
        if smoke
        else "dinov2_vits14",
    }
    if kind == "lewam":
        if rollout_steps != 1:
            raise ValueError("LeWAM uses chunk_size for multi-step dynamics; leave rollout_steps=1")
        encoder_dim = encoder_dim if encoder is not None else 32
        if encoder_dim is None:
            raise ValueError("A custom encoder requires encoder_dim")
        recipe.update(chunk_size=chunk_size, action_scale=list(action_scale), proprioception=proprioception)
        if command_state:
            recipe["command_state"] = True
        model = LeWAM(
            encoder if encoder is not None else observation_encoder((3, size, size)),
            4,
            embed_dim=encoder_dim,
            history=history,
            chunk_size=chunk_size,
            dim=64 if smoke else 128,
            depth=2 if smoke else 4,
            num_heads=4,
            mlp_dim=128 if smoke else 512,
            flow_steps=8,
            state_dim=(10 if command_state else 6) if proprioception else 0,
            action_scale=action_scale,
        )
    elif kind == "lewm":
        if proprioception:
            recipe["proprioception"] = True
        if rollout_steps != 1:
            recipe["rollout_steps"] = rollout_steps
        encoder_dim = encoder_dim if encoder is not None else 32
        if encoder_dim is None:
            raise ValueError("A custom encoder requires encoder_dim")
        model = LeWM(
            encoder if encoder is not None else observation_encoder((3, size, size)),
            4,
            embed_dim=encoder_dim,
            history=history,
            depth=2,
            num_heads=4,
            head_dim=16,
            mlp_dim=128,
            proj_dim=64,
            dropout=0,
            state_dim=6 if proprioception else 0,
            rollout_steps=rollout_steps,
        )
    else:
        if encoder is None:
            patch = 4 if smoke else 14
            if size % patch:
                raise ValueError(f"Image size must be divisible by patch size {patch}")
            encoder = PixelPatches(patch) if smoke else PatchTokens.from_hub(size=size)
            encoder_dim, grid = encoder.dim, (size // patch, size // patch)
        if encoder_dim is None or grid is None or len(grid) != 2 or min(grid) < 1:
            raise ValueError("Custom patch encoders require encoder_dim and a two-dimensional grid")
        recipe.update(grid=list(grid), encoder_dim=encoder_dim)
        if kind == "dinowm":
            model = DINOWM(
                encoder,
                grid[0] * grid[1],
                encoder_dim,
                4,
                proprio_dim=6,
                history=history,
                depth=2,
                num_heads=4,
                head_dim=16,
                mlp_dim=128,
                dropout=0,
            )
        else:
            model = RoboJEPA(
                encoder_dim,
                4,
                state_dim=6,
                num_views=1,
                predictor_dim=64,
                depth=2,
                num_heads=4,
                window=history,
                rollout_steps=min(2, history),
                encoder=encoder,
            )
    return ModelAdapter(model, recipe)


class WorldModelPolicy:
    """CEM or LeWAM proposals with observations and actually applied past actions.

    Call ``record_action`` after every environment step. This matters when joint
    limits, actuator delay, or the hardware driver change a requested command.
    """

    def __init__(self, adapter, goal, *, planner=None, seed=7):
        self.adapter = adapter.eval()
        self.planner = planner or (
            LeWAMPlanner()
            if adapter.kind == "lewam"
            else CEM(8, 4, num_samples=64, num_elites=8, iterations=3, bounds=(-1, 1))
        )
        self.seed = seed
        self.goal = goal
        self.reset()
        with torch.no_grad():
            self.goal_latent = adapter.encode(
                goal.rgb[None, None].to(adapter.device), adapter.observation_state(goal)[None, None].to(adapter.device)
            )[:, 0]

    def reset(self):
        self.observations = deque(maxlen=self.adapter.history)
        self.actions = deque(maxlen=max(1, self.adapter.history - 1))
        self.latencies = []
        self.pending = False
        self.generator = torch.Generator(device=self.adapter.device).manual_seed(self.seed)

    @torch.no_grad()
    def __call__(self, observation):
        if self.pending:
            raise RuntimeError("Call record_action with the applied action before planning the next step")
        if observation.rgb.shape != self.goal.rgb.shape:
            raise ValueError("Observation and goal image shapes differ")
        self.observations.append(observation)
        pixels = torch.stack([o.rgb for o in self.observations])[None].to(self.adapter.device)
        states = torch.stack([self.adapter.observation_state(o) for o in self.observations])[None].to(
            self.adapter.device
        )
        past = (
            torch.stack(list(self.actions)[-(len(self.observations) - 1) :])[None].to(self.adapter.device)
            if len(self.observations) > 1
            else None
        )
        # Match LeWAMWindows' initial-history padding, including CEM's optional
        # past-action contract. The learned LeWAM proposal planner ignores past.
        if self.adapter.kind == "lewam" and pixels.size(1) < self.adapter.history:
            missing = self.adapter.history - pixels.size(1)
            pixels = torch.cat((pixels[:, :1].expand(-1, missing, -1, -1, -1), pixels), dim=1)
            states = torch.cat((states[:, :1].expand(-1, missing, -1), states), dim=1)
            if past is None:
                past = states.new_zeros(1, 0, 4)
            past = torch.cat((past.new_zeros(1, missing, 4), past), dim=1)
        started = time.perf_counter()
        latent = self.adapter.encode(pixels, states)
        plan = self.planner(
            self.adapter.model,
            latent,
            self.goal_latent,
            past_actions=past,
            generator=self.generator,
            **self.adapter.conditions(states),
        )
        action = plan[0, 0].cpu()
        self.latencies.append(time.perf_counter() - started)
        self.pending = True
        return action

    def record_action(self, applied):
        if not self.pending:
            raise RuntimeError("No pending policy action")
        action = torch.as_tensor(applied, dtype=torch.float32).detach().cpu()
        if action.shape != (4,) or not torch.isfinite(action).all():
            raise ValueError("Expected finite applied action (4,)")
        self.actions.append(action)
        self.pending = False

    def check_latency(self, observation, control_dt, *, repeats=3):
        """Motor-free preflight; reject a planner slower than the control interval."""
        latencies = []
        for _ in range(repeats):
            self.reset()
            for _ in range(self.adapter.history):
                self(observation)
                latencies.append(self.latencies[-1])
                self.record_action(torch.zeros(4))
        self.reset()
        if max(latencies) > control_dt:
            raise TimeoutError(f"Planner needs {max(latencies):.3f}s, exceeding control_dt={control_dt:.3f}s")
        return latencies
