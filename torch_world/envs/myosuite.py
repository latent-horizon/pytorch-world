"""MyoSuite musculoskeletal tasks behind the batched benchmark protocol: one upstream environment per lane."""

from torch_world.envs.lanes import MujocoLanes, suite_spec

LOCOMOTION = ("Walk", "Directional", "Run")
"""Name parts of the locomotion tasks, whose ``solved`` flag tracks a commanded velocity at each step."""


class MyoSuite(MujocoLanes):
    """A `MyoSuite <https://github.com/MyoHub/myosuite>`_ musculoskeletal task, by its registered name.

    ``name`` is any MyoSuite environment, such as ``"myoHandPoseRandom-v0"``, ``"myoHandReorient8-v0"`` or
    ``"myoLegWalk-v0"``. Actions are muscle activations, normalized by MyoSuite to [-1, 1]: one per muscle, 39 for
    the hand. Observations are MyoSuite's state vector ``(B, D)``, which includes each task's target, or with
    ``observation="rgb"`` images ``(B, 3, size, size)`` in [0, 1] from ``camera``: by default the model's first named
    camera, or MuJoCo's free camera when it has none.

    Rewards are MyoSuite's dense task rewards; ``transition.success`` is its ``solved`` flag, and with
    ``terminate_on_success`` the episode ends there. By default it does, except in the locomotion tasks, whose
    ``solved`` means tracking the commanded velocity at that step rather than reaching a goal. MyoSuite's own
    terminations, such as a fall, also end the episode. The episode limit is the task's registered one.
    ``**env_kwargs`` go to MyoSuite's constructor.
    """

    suite = "MyoSuite"

    def __init__(
        self,
        name,
        *,
        batch_size=1,
        observation="state",
        size=None,
        camera=None,
        terminate_on_success=None,
        quality="high",
        render_size=None,
        device=None,
        max_episode_steps=None,
        **env_kwargs,
    ):
        if terminate_on_success is None:
            terminate_on_success = not any(part in name for part in LOCOMOTION)
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
            f"MyoSuite/{name}",
            "torch_world.envs.myosuite:MyoSuite",
            "myosuite",
            ("myosuite",),
            {"name": name},
            "https://github.com/MyoHub/myosuite",
        )
        super().__init__(
            batch_size, size, device, observation=observation, camera=camera, quality=quality, render_size=render_size
        )
        try:
            from myosuite.utils import gym
        except ImportError as exc:
            raise ImportError("Install MyoSuite with: pip install 'torch-world[myosuite]'") from exc
        if name not in gym.envs.registry:
            raise ValueError(f"Unknown MyoSuite environment {name!r}; tasks are named like 'myoHandPoseRandom-v0'")
        if max_episode_steps is not None:
            env_kwargs["max_episode_steps"] = max_episode_steps
        self._build(lambda: gym.make(name, **env_kwargs))
        if camera is None:
            model = self._physics(0)[0]
            self.camera = model.camera(0).name if model.ncam else -1
        self._bounds(self.envs[0].action_space.low, self.envs[0].action_space.high)
        self.max_episode_steps = max_episode_steps or self.envs[0].spec.max_episode_steps
        self.reset(seed=0)

    def _physics(self, lane):
        env = self.envs[lane].unwrapped
        return env.model, env.data

    def _reset_lane(self, lane, seed):
        return self.envs[lane].reset(seed=seed)[0]

    def _step_lane(self, lane, action):
        state, reward, terminated, truncated, info = self.envs[lane].step(action)
        success = bool(info["solved"])
        return state, reward, terminated or (success and self.terminate_on_success), truncated, success
