"""Shared parts of the external MuJoCo suites: one upstream environment per lane behind the batched protocol."""

import copy

import numpy as np
import torch

from torch_world.envs.benchmark import BenchmarkEnv, Transition, seed_from_generator
from torch_world.envs.mujoco import import_mujoco
from torch_world.envs.registry import EnvSpec, get_spec


def suite_spec(id, entry_point, backend, tags, kwargs, reference):
    """The catalog spec of ``id``, or one built for a task outside the catalog."""
    try:
        return get_spec(id)
    except KeyError:
        return EnvSpec(id, entry_point, backend, id.replace("/", " "), tags, kwargs, reference)


class MujocoLanes(BenchmarkEnv):
    """Stepping and rendering for suites that run one upstream MuJoCo environment per lane.

    A subclass builds ``self.envs`` with ``_build``, sets the upstream action bounds with ``_bounds`` and implements
    ``_physics``, ``_reset_lane`` and ``_step_lane``. Actions in [-1, 1] are rescaled to the bounds; finished lanes
    are absorbing and never stepped. Frames come from a MuJoCo renderer per lane, so each shows its own goal markers.

    ``observation="state"`` observes the upstream state vector ``(B, D)``; ``"rgb"`` observes ``camera`` images
    ``(B, 3, size, size)`` in [0, 1]. ``state`` is always the vector: states are defined in the simulator, so they
    cannot be restored or scored from tensors.
    """

    suite = "upstream"
    has_success = True

    def __init__(self, batch_size, size, device, *, observation, camera, flip=False, quality, render_size):
        if observation not in ("state", "rgb"):
            raise ValueError("observation must be 'state' or 'rgb'")
        self.mujoco = import_mujoco()  # before the suite loads MuJoCo, so headless Linux renders with EGL
        # The episode limit comes from the upstream task, read once the environments exist.
        super().__init__(batch_size, size, device, 1, quality=quality, render_size=render_size)
        self.observation, self.camera, self.flip = observation, camera, flip
        self.envs, self._renderers, self._start = [], {}, None

    def _build(self, make):
        try:
            for _ in range(self.batch_size):
                self.envs.append(make())
        except Exception:
            self.close()
            raise

    def _bounds(self, low, high):
        self._low, self._high = (torch.as_tensor(np.asarray(b, dtype=np.float32).ravel()) for b in (low, high))
        self.action_dim = self._low.numel()

    def _physics(self, lane):
        """The lane's ``(mujoco.MjModel, mujoco.MjData)``."""
        raise NotImplementedError

    def _reset_lane(self, lane, seed):
        """Reset one lane with ``seed`` and return its state vector."""
        raise NotImplementedError

    def _step_lane(self, lane, action):
        """Apply an upstream action; return ``(state, reward, terminated, truncated, success or None)``."""
        raise NotImplementedError

    def _pixels(self, lanes, size):
        """Frames ``(len(lanes), 3, size, size)`` as uint8 from each lane's ``camera``."""
        frames = []
        for lane in lanes:
            model, data = self._physics(lane)
            if (lane, size) not in self._renderers:
                buffer = model.vis.global_
                buffer.offwidth, buffer.offheight = max(buffer.offwidth, size), max(buffer.offheight, size)
                self._renderers[lane, size] = self.mujoco.Renderer(model, size, size)
            renderer = self._renderers[lane, size]
            renderer.update_scene(data, self.camera)
            image = renderer.render()
            frames.append(image[::-1] if self.flip else image)
        return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()

    def _observe(self, lanes):
        if self.observation == "state":
            return self.state[lanes].clone()
        return self._pixels(lanes, self.size).to(self.device).float() / 255

    def _snapshot(self, lane):
        model, data = self._physics(lane)
        kind = self.mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(self.mujoco.mj_stateSize(model, kind))
        self.mujoco.mj_getState(model, data, state, kind)
        return state

    def reset(self, state=None, *, seed=None, generator=None):
        if state is not None:
            raise ValueError(f"{self.suite} observations are not simulator checkpoints; use reset(seed=...)")
        if seed is not None and generator is not None:
            raise ValueError("Pass seed or generator, not both")
        seed = seed_from_generator(generator) if seed is None else seed
        self._reset_episode()
        lanes = list(range(self.batch_size))
        states = [torch.as_tensor(np.asarray(self._reset_lane(lane, seed + lane), dtype=np.float32)) for lane in lanes]
        self.state = torch.stack(states).to(self.device)
        self.state_dim = self.state.shape[-1]
        self._obs = self._observe(lanes)
        self._success = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self._start = [self._snapshot(lane) for lane in lanes]
        return self.render()

    def step(self, action):
        action = self._action(action)
        active = ~self.transition.done
        action = (self._low + (action.cpu() + 1) / 2 * (self._high - self._low)).numpy()
        state, reward = self.state.clone(), torch.zeros(self.batch_size, device=self.device)
        terminated, truncated = torch.zeros_like(active), torch.zeros_like(active)
        success = self._success.clone()
        lanes = active.nonzero().flatten().tolist()
        for lane in lanes:
            lane_state, r, term, trunc, solved = self._step_lane(lane, action[lane])
            state[lane] = torch.as_tensor(np.asarray(lane_state, dtype=np.float32), device=self.device)
            reward[lane], terminated[lane], truncated[lane] = float(r), bool(term), bool(trunc)
            if solved is not None:
                success[lane] = bool(solved)
        self.state, self._success = state, success
        if lanes:
            self._obs[lanes] = self._observe(lanes)
        self.elapsed += active
        self.transition = Transition(
            reward,
            self.transition.terminated | terminated,
            self.transition.truncated | truncated | ((self.elapsed >= self.max_episode_steps) & active),
            success.clone() if self.has_success else None,
        )
        return self.render()

    def render(self, state=None):
        """The current observation; states live in the simulator, so they cannot be rendered from tensors."""
        if state is not None:
            raise ValueError(f"Arbitrary-state rendering is not supported by {self.suite}")
        return self._obs.clone()

    def render_frame(self, state=None):
        """Frames ``(B, 3, render_size, render_size)`` as uint8 from ``camera``, of the simulator's current state."""
        if state is not None:
            raise ValueError(f"Arbitrary-state rendering is not supported by {self.suite}")
        return self._pixels(range(self.batch_size), self.render_size)

    def render_goal_frame(self):
        """The scene at reset: these suites define goals in the simulator, not as an observation.

        Each lane is posed at its reset state for the frame, then restored exactly; no physics step runs.
        """
        if self._start is None:
            raise RuntimeError("Call reset() before rendering the reset scene")
        saved = []
        try:
            for lane, start in enumerate(self._start):
                model, data = self._physics(lane)
                saved.append(copy.copy(data))
                self.mujoco.mj_setState(model, data, start, self.mujoco.mjtState.mjSTATE_INTEGRATION)
                self.mujoco.mj_forward(model, data)
            return self.render_frame()
        finally:
            for lane, data in enumerate(saved):
                model, current = self._physics(lane)
                self.mujoco.mj_copyData(current, model, data)

    @property
    def gt_label(self):
        return "Task at reset"

    def close(self):
        for renderer in getattr(self, "_renderers", {}).values():
            renderer.close()
        self._renderers = {}
        for env in getattr(self, "envs", ()):
            env.close()
        self.envs = []
