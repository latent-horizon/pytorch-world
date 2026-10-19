"""DeepMind Control Suite tasks behind the batched benchmark protocol: one upstream environment per lane."""

import numpy as np

from torch_world.envs.lanes import MujocoLanes, suite_spec

CAMERAS = {"quadruped": 2}
"""Domains whose task camera is not camera 0, as in DreamerV3 and TD-MPC2."""


class DMControl(MujocoLanes):
    """A `DeepMind Control Suite <https://github.com/google-deepmind/dm_control>`_ task, by ``"domain-task"`` name.

    ``name`` is any suite task, such as ``"cheetah-run"``, ``"walker-walk"`` or ``"ball_in_cup-catch"``.
    Observations are the suite's observation dictionary flattened into one vector ``(B, D)``, or with
    ``observation="rgb"`` images ``(B, 3, size, size)`` in [0, 1] from ``camera``, 64 x 64 by default as in DreamerV3
    and TD-MPC2. ``action_repeat`` applies each action for that many control steps and sums their rewards, which
    divides the 1,000-step episode limit.

    Rewards are the suite's, in [0, 1] per control step. The suite has no success condition, so
    ``transition.success`` is ``None``; episodes end at the time limit. ``**task_kwargs`` go to the task, such as
    ``time_limit``; its ``random`` seed is set by ``reset(seed=...)``, ``seed + lane`` per lane.
    """

    suite = "DMControl"
    has_success = False

    def __init__(
        self,
        name,
        *,
        batch_size=1,
        observation="state",
        size=None,
        camera=None,
        action_repeat=1,
        quality="high",
        render_size=None,
        device=None,
        max_episode_steps=None,
        **task_kwargs,
    ):
        domain, _, task = name.partition("-")
        if action_repeat < 1:
            raise ValueError("action_repeat must be positive")
        self.name, self.action_repeat = name, action_repeat
        self.settings = {
            "name": name,
            "batch_size": batch_size,
            "observation": observation,
            "size": size,
            "camera": camera,
            "action_repeat": action_repeat,
            "quality": quality,
            "render_size": render_size,
            "max_episode_steps": max_episode_steps,
            **task_kwargs,
        }
        self.spec = suite_spec(
            f"DMControl/{name}",
            "torch_world.envs.dmcontrol:DMControl",
            "dmcontrol",
            ("dmcontrol",),
            {"name": name},
            "https://github.com/google-deepmind/dm_control",
        )
        camera = CAMERAS.get(domain, 0) if camera is None else camera
        size = 64 if size is None and observation == "rgb" else size
        super().__init__(
            batch_size, size, device, observation=observation, camera=camera, quality=quality, render_size=render_size
        )
        try:
            from dm_control import suite
        except ImportError as exc:
            raise ImportError("Install the DeepMind Control Suite with: pip install 'torch-world[dmcontrol]'") from exc
        if (domain, task) not in suite.ALL_TASKS:
            raise ValueError(f"Unknown DMControl task {name!r}; names are 'domain-task', such as 'cheetah-run'")
        self._build(lambda: suite.load(domain, task, task_kwargs={"random": 0, **task_kwargs}))
        spec = self.envs[0].action_spec()
        self._bounds(spec.minimum, spec.maximum)
        steps = self.envs[0]._step_limit
        self.max_episode_steps = max_episode_steps or int(np.ceil(steps / action_repeat))
        self.reset(seed=0)

    def _physics(self, lane):
        physics = self.envs[lane].physics
        return physics.model.ptr, physics.data.ptr

    @staticmethod
    def _vector(observation):
        return np.concatenate([np.asarray(value, dtype=np.float32).ravel() for value in observation.values()])

    def _reset_lane(self, lane, seed):
        env = self.envs[lane]
        env.task._random = np.random.RandomState(seed)  # the suite draws initial states only from the task's stream
        return self._vector(env.reset().observation)

    def _step_lane(self, lane, action):
        reward = 0.0
        for _ in range(self.action_repeat):
            step = self.envs[lane].step(action)
            reward += step.reward or 0.0
            if step.last():
                break
        # A terminal discount of 0 is a task's own termination; otherwise the episode reached its time limit.
        terminated = step.last() and step.discount == 0
        return self._vector(step.observation), reward, terminated, step.last() and not terminated, None
