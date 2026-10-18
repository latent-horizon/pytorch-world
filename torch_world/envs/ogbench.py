"""OGBench tasks behind the batched benchmark protocol: one upstream Gymnasium environment per lane."""

import math

import numpy as np
import torch

from torch_world.envs.benchmark import BenchmarkEnv, Transition, seed_from_generator
from torch_world.envs.registry import EnvSpec, get_spec
from torch_world.envs.utils import NumpyStreams


class OGBench(BenchmarkEnv):
    """An `OGBench <https://github.com/seohongpark/ogbench>`_ task, by its dataset name.

    ``name`` is any OGBench dataset name, such as ``"cube-single-play-v0"``, ``"visual-antmaze-medium-navigate-v0"``
    or ``"scene-play-singletask-task2-v0"``; the dataset type only selects the data, not the environment.

    Goal-conditioned names evaluate one of the five evaluation tasks, ``task_id``. The goal is an observation,
    ``goal``, so ``render(goal)`` returns it and planners reach its embedding; the episode ends when OGBench reports
    ``success``. Single-task names have no ``goal`` and keep OGBench's rewards, from ``-num_subtasks`` up to 0.

    Observations are OGBench's state vectors ``(B, D)``, or ``visual-`` RGB images ``(B, 3, size, size)`` in [0, 1],
    64 x 64 by default as in the OGBench datasets. ``state`` is the observation: OGBench defines goals and success in
    the simulator, so states cannot be restored or scored from tensors.

    OGBench also draws from NumPy's global generator, so each lane keeps its own global stream and seeded resets are
    reproducible. One exception is upstream: manipulation goals settle under two random actions from an unseeded
    action space, so the arm in the goal varies between resets; the object targets, which define success, do not.
    Discrete-action powderworld tasks are not supported.
    """

    def __init__(
        self,
        name,
        *,
        task_id=1,
        batch_size=1,
        size=None,
        quality="high",
        render_size=None,
        device=None,
        max_episode_steps=None,
        **env_kwargs,
    ):
        try:
            import ogbench
        except ImportError as exc:
            raise ImportError("Install OGBench with: pip install 'torch-world[ogbench]'") from exc
        self.name, self.visual = name, name.startswith("visual-")
        self.goal_conditioned = "-singletask" not in name
        if self.goal_conditioned and task_id not in range(1, 6):
            raise ValueError("task_id must be one of OGBench's evaluation tasks, 1 to 5")
        if not self.goal_conditioned and task_id != 1:
            raise ValueError("Single-task names fix the task; select it in the name, e.g. '-singletask-task2-v0'")
        self.task_id = task_id
        self.settings = {
            "name": name,
            "task_id": task_id,
            "batch_size": batch_size,
            "size": size,
            "quality": quality,
            "render_size": render_size,
            "max_episode_steps": max_episode_steps,
            **env_kwargs,
        }
        try:
            self.spec = get_spec(f"OGBench/{name}")
        except KeyError:  # an OGBench task outside the catalog carries its own spec
            self.spec = EnvSpec(
                f"OGBench/{name}",
                "torch_world.envs.ogbench:OGBench",
                "ogbench",
                f"OGBench {name}",
                ("ogbench", "goal") if self.goal_conditioned else ("ogbench",),
                {"name": name},
                "https://github.com/seohongpark/ogbench",
            )
        size = 64 if size is None and self.visual else size
        # The episode limit comes from the upstream registration, read once the environments exist.
        super().__init__(batch_size, size, device, max_episode_steps or 1, quality=quality, render_size=render_size)
        # Visual tasks observe through the camera at `size`; state tasks only render presentation frames.
        resolution = self.size if self.visual else self.render_size
        env_kwargs = {"width": resolution, "height": resolution, **env_kwargs}
        if max_episode_steps is not None:
            env_kwargs["max_episode_steps"] = max_episode_steps
        self.envs = []
        try:
            for _ in range(batch_size):
                self.envs.append(ogbench.make_env_and_datasets(name, env_only=True, **env_kwargs))
        except Exception:
            self.close()
            raise
        space = self.envs[0].action_space
        if not hasattr(space, "low"):
            self.close()
            raise ValueError(f"{name} has discrete actions; OGBench adapters support continuous control")
        self._low, self._high = (torch.as_tensor(b, dtype=torch.float32) for b in (space.low, space.high))
        self.action_dim = space.shape[0]
        self.max_episode_steps = self.envs[0].spec.max_episode_steps
        self.observation_shape = (3, self.size, self.size) if self.visual else self.envs[0].observation_space.shape
        self.state_dim = math.prod(self.observation_shape)
        self._streams = NumpyStreams(batch_size)
        self._goal_frame = None
        self._reset_episode()

    def _observation(self, observations):
        obs = torch.as_tensor(np.stack(observations), device=self.device)
        return obs.permute(0, 3, 1, 2).float() / 255 if self.visual else obs.float()

    @staticmethod
    def _frame(images):
        return torch.as_tensor(np.stack(images)).permute(0, 3, 1, 2).contiguous()

    def reset(self, state=None, *, seed=None, generator=None):
        if state is not None:
            raise ValueError("OGBench observations are not simulator checkpoints; use reset(seed=...)")
        if seed is not None and generator is not None:
            raise ValueError("Pass seed or generator, not both")
        seed = seed_from_generator(generator) if seed is None else seed
        self._reset_episode()
        options = {"render_goal": True}
        if self.goal_conditioned:
            options["task_id"] = self.task_id
        observations, goals, frames = [], [], []
        for lane, env in enumerate(self.envs):
            self._streams.seed(lane, seed + lane)
            obs, info = self._streams.call(lane, env.reset, seed=seed + lane, options=options)
            observations.append(obs)
            goals.append(info.get("goal"))
            frames.append(info.get("goal_rendered"))
        self._obs = self._observation(observations)
        self.state = self._obs.clone()
        self._success = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self._goal_frame = self._frame(frames) if all(f is not None for f in frames) else None
        if self.goal_conditioned:
            self.goal = self._observation(goals)
        return self.render()

    def step(self, action):
        action = self._action(action)
        active = ~self.transition.done
        action[~active] = 0
        action = (self._low + (action.cpu() + 1) / 2 * (self._high - self._low)).numpy()
        obs, reward = self._obs.clone(), torch.zeros(self.batch_size, device=self.device)
        terminated, truncated = torch.zeros_like(active), torch.zeros_like(active)
        success = self._success.clone()
        for lane in active.nonzero().flatten().tolist():
            observation, r, term, trunc, info = self._streams.call(lane, self.envs[lane].step, action[lane])
            obs[lane] = self._observation([observation])[0]
            reward[lane], terminated[lane], truncated[lane] = float(r), bool(term), bool(trunc)
            success[lane] = bool(info["success"])
        self._obs, self.state, self._success = obs, obs.clone(), success
        self.elapsed += active
        self.transition = Transition(
            reward,
            self.transition.terminated | terminated,
            self.transition.truncated | truncated | ((self.elapsed >= self.max_episode_steps) & active),
            success.clone(),
        )
        return self.render()

    def render(self, state=None):
        """The current observation, or ``state`` itself: OGBench states and goals are observations."""
        if state is None:
            return self._obs.clone()
        state = torch.as_tensor(state, device=self.device)
        if state.shape[1:] != self.observation_shape:
            raise ValueError(f"Expected observations of shape (B, {', '.join(map(str, self.observation_shape))})")
        return state.clone()

    def render_frame(self, state=None):
        """Upstream camera frames ``(B, 3, H, W)`` as uint8, from the simulator's current state."""
        if state is not None:
            raise ValueError("Arbitrary-state rendering is not supported by OGBench")
        return self._frame([self._streams.call(lane, env.render) for lane, env in enumerate(self.envs)])

    def render_goal_frame(self):
        """OGBench's rendering of the goal state at reset, from the same camera as ``render_frame``."""
        return self._goal_frame.clone() if self._goal_frame is not None else self.render_frame()

    def close(self):
        for env in getattr(self, "envs", ()):
            env.close()
        self.envs = []
