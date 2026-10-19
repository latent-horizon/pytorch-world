"""Meta-World tasks behind the batched benchmark protocol: one upstream environment per lane."""

from torch_world.envs.lanes import MujocoLanes, suite_spec


class MetaWorld(MujocoLanes):
    """A `Meta-World <https://github.com/Farama-Foundation/Metaworld>`_ v3 task with a Sawyer arm, by name.

    ``name`` is any of the 50 v3 tasks, such as ``"reach-v3"``, ``"pick-place-v3"`` or ``"door-open-v3"``. Each
    reset samples a new object and goal placement, as Meta-World's MT1 does, from the lane's own generator, so
    ``reset(seed=...)`` is reproducible. The goal position is observed: observations are Meta-World's 39-dimensional
    goal-observable state ``(B, 39)``, or with ``observation="rgb"`` images ``(B, 3, size, size)`` in [0, 1] from
    ``camera``.

    Actions are end-effector position deltas and the gripper, 4 in [-1, 1]. Rewards are Meta-World's dense v2 shaping;
    ``transition.success`` is its ``success`` flag, and with ``terminate_on_success`` the episode ends there, as in
    Meta-World's evaluation. Episodes last 500 steps. Meta-World's corner cameras render upside down; their frames
    are flipped upright.
    """

    suite = "Meta-World"

    def __init__(
        self,
        name,
        *,
        batch_size=1,
        observation="state",
        size=None,
        camera="corner2",
        terminate_on_success=True,
        quality="high",
        render_size=None,
        device=None,
        max_episode_steps=None,
        **env_kwargs,
    ):
        self.name, self.terminate_on_success = name, terminate_on_success
        self.settings = {
            "name": name,
            "batch_size": batch_size,
            "observation": observation,
            "size": size,
            "camera": camera,
            "terminate_on_success": terminate_on_success,
            "quality": quality,
            "render_size": render_size,
            "max_episode_steps": max_episode_steps,
            **env_kwargs,
        }
        self.spec = suite_spec(
            f"MetaWorld/{name}",
            "torch_world.envs.metaworld:MetaWorld",
            "metaworld",
            ("metaworld",),
            {"name": name},
            "https://github.com/Farama-Foundation/Metaworld",
        )
        flip = isinstance(camera, str) and camera.startswith("corner")
        super().__init__(
            batch_size,
            size,
            device,
            observation=observation,
            camera=camera,
            flip=flip,
            quality=quality,
            render_size=render_size,
        )
        try:
            from metaworld.env_dict import ALL_V3_ENVIRONMENTS
        except ImportError as exc:
            raise ImportError("Install Meta-World with: pip install 'torch-world[metaworld]'") from exc
        if name not in ALL_V3_ENVIRONMENTS:
            raise ValueError(f"Unknown Meta-World task {name!r}; tasks are named like 'pick-place-v3'")

        def make():
            env = ALL_V3_ENVIRONMENTS[name](**env_kwargs)
            # The goal-observable MT1 setup: every reset samples a placement, from the environment's seeded generator.
            env._partially_observable = False
            # Steps clip observations to this space, cached when the goal was hidden; recompute it with the goal.
            env.__dict__.pop("sawyer_observation_space", None)
            env._freeze_rand_vec = False
            env.seeded_rand_vec = True
            env._set_task_called = True
            if max_episode_steps is not None:
                env.max_path_length = max_episode_steps
            return env

        self._build(make)
        self._bounds(self.envs[0].action_space.low, self.envs[0].action_space.high)
        self.max_episode_steps = self.envs[0].max_path_length
        self.reset(seed=0)

    def _physics(self, lane):
        return self.envs[lane].model, self.envs[lane].data

    def _reset_lane(self, lane, seed):
        env = self.envs[lane]
        env.seed(seed)
        return env.reset()[0]

    def _step_lane(self, lane, action):
        state, reward, terminated, truncated, info = self.envs[lane].step(action)
        success = bool(info["success"])
        return state, reward, terminated or (success and self.terminate_on_success), truncated, success
