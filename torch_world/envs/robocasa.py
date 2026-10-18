"""RoboCasa kitchen tasks behind the batched benchmark protocol: one upstream robosuite environment per lane."""

import sys

import torch

from torch_world.envs.benchmark import BenchmarkEnv, Transition, seed_from_generator
from torch_world.envs.registry import EnvSpec, get_spec
from torch_world.envs.utils import NumpyStreams

PRESENTATION_CAMERA = "robot0_agentview_center"


class RoboCasa(BenchmarkEnv):
    """A `RoboCasa <https://github.com/robocasa/robocasa>`_ kitchen task, by its registered name.

    ``task`` is any RoboCasa task, atomic (``"OpenDrawer"``) or composite (``"PrepareCoffee"``), performed by the
    mobile ``PandaOmron`` robot. ``split="target"`` samples RoboCasa's 10 target kitchens and held-out object
    instances, ``"pretrain"`` its 2,500 pretraining scenes. Tasks are specified in language, ``instruction`` per
    lane, so there is no goal observation: episodes end when RoboCasa's own success check passes, with reward 1, or
    at the task's horizon from RoboCasa's dataset registry.

    Observations are ``camera`` frames ``(B, 3, size, size)`` in [0, 1], flipped upright, or with
    ``observation="state"`` the robot's proprioception ``(B, D)``. ``state`` is always that vector. Object poses are
    left out: their number depends on the sampled kitchen, so lanes would not stack. Actions are RoboCasa's 12 controls in [-1, 1]: end-effector position and rotation deltas, gripper,
    base motion, torso and the base/arm mode switch, in that order.

    Every reset rebuilds the kitchen, which takes seconds per lane. On macOS, robosuite's GLFW contexts corrupt the
    images of every RoboCasa environment in a process but the last, so batched pixel observations are refused there;
    presentation frames of batched state observations are affected too. RoboCasa pins ``mujoco==3.3.1``, so it runs in its
    own environment; see the benchmark guide.
    """

    def __init__(
        self,
        task,
        *,
        split="target",
        batch_size=1,
        size=128,
        camera="robot0_agentview_left",
        observation="rgb",
        quality="high",
        render_size=None,
        device=None,
        max_episode_steps=None,
        robot="PandaOmron",
        **env_kwargs,
    ):
        try:
            from robocasa.utils.dataset_registry_utils import get_task_horizon
            from robocasa.utils.env_utils import create_env
        except ImportError as exc:
            raise ImportError("RoboCasa is installed from source; see docs/guides/benchmark-environments.md") from exc
        if observation not in ("rgb", "state"):
            raise ValueError("observation must be 'rgb' or 'state'")
        if sys.platform == "darwin" and batch_size > 1 and observation == "rgb":
            raise ValueError(
                "On macOS, robosuite's GLFW contexts corrupt the images of all but the last RoboCasa environment in a "
                "process; use batch_size=1, observation='state', or Linux with MUJOCO_GL=egl"
            )
        self.task, self.split, self.camera, self.observation = task, split, camera, observation
        self.settings = {
            "task": task,
            "split": split,
            "batch_size": batch_size,
            "size": size,
            "camera": camera,
            "observation": observation,
            "quality": quality,
            "render_size": render_size,
            "max_episode_steps": max_episode_steps,
            "robot": robot,
            **env_kwargs,
        }
        try:
            self.spec = get_spec(f"RoboCasa/{task}")
        except KeyError:  # a RoboCasa task outside the catalog carries its own spec
            self.spec = EnvSpec(
                f"RoboCasa/{task}",
                "torch_world.envs.robocasa:RoboCasa",
                "robocasa",
                f"RoboCasa {task}",
                ("robocasa",),
                {"task": task},
                "https://robocasa.ai",
            )
        if max_episode_steps is None:
            max_episode_steps = get_task_horizon(task)
        super().__init__(batch_size, size, device, max_episode_steps, quality=quality, render_size=render_size)
        # Presentation frames come from the observation pipeline, rendered during each lane's own reset and step:
        # rendering a lane later, after another lane rebuilt its kitchen, can return corrupted frames.
        cameras = {camera: size, PRESENTATION_CAMERA: self.render_size}
        self.envs = []
        try:
            for _ in range(batch_size):
                self.envs.append(
                    create_env(
                        task,
                        robots=robot,
                        camera_names=list(cameras),
                        camera_widths=list(cameras.values()),
                        camera_heights=list(cameras.values()),
                        split=split,
                        seed=0,
                        **env_kwargs,
                    )
                )
        except Exception:
            self.close()
            raise
        self._streams = NumpyStreams(batch_size)
        self.instruction = [""] * batch_size
        self.reset(seed=0)  # robosuite builds the robot, and its action limits, at the first reset
        low, high = self.envs[0].action_spec
        self._low, self._high = torch.as_tensor(low, dtype=torch.float32), torch.as_tensor(high, dtype=torch.float32)
        self.action_dim = len(low)

    def _vector(self, obs):
        return torch.as_tensor(obs["robot0_proprio-state"], dtype=torch.float32)

    @staticmethod
    def _pixels(obs, camera):
        """A camera image, flipped upright from robosuite's OpenGL convention, as uint8 ``(3, H, W)``."""
        return torch.as_tensor(obs[f"{camera}_image"][::-1].copy()).permute(2, 0, 1)

    def _read(self, observations):
        state = torch.stack([self._vector(o) for o in observations]).to(self.device)
        frames = torch.stack([self._pixels(o, PRESENTATION_CAMERA) for o in observations])
        if self.observation == "state":
            return state, state.clone(), frames
        images = torch.stack([self._pixels(o, self.camera) for o in observations]).float()
        if images.shape[-1] != self.size:  # the observation camera is the presentation camera, at render_size
            images = torch.nn.functional.interpolate(images, (self.size, self.size), mode="area")
        return (images / 255).to(self.device), state, frames

    def reset(self, state=None, *, seed=None, generator=None, scenes=None):
        """Sample new kitchens, or with ``scenes`` start each lane from a recorded demonstration.

        ``scenes`` holds one `load_robocasa_scene` per lane: the demonstration's kitchen, objects and initial
        simulator state, restored as RoboCasa's playback script does, so the demonstration's frames are reachable
        goals.
        """
        if state is not None:
            raise ValueError("RoboCasa observations are not simulator checkpoints; use reset(seed=...) or scenes")
        if seed is not None and generator is not None:
            raise ValueError("Pass seed or generator, not both")
        if scenes is not None and len(scenes) != self.batch_size:
            raise ValueError(f"Pass one scene per lane, {self.batch_size}")
        import numpy as np

        seed = seed_from_generator(generator) if seed is None else seed
        self._reset_episode()
        observations = []
        for lane, env in enumerate(self.envs):
            # The kitchen, objects and robot noise come from env.rng; counter colors from NumPy's global generator.
            env.rng = np.random.default_rng(seed + lane)
            self._streams.seed(lane, seed + lane)
            if scenes is None:
                observations.append(self._streams.call(lane, env.reset))
            else:
                observations.append(self._streams.call(lane, self._restore, env, scenes[lane]))
            self.instruction[lane] = env.get_ep_meta().get("lang", "")
        self._obs, self.state, self._frames = self._read(observations)
        self.state_dim = self.state.shape[-1]
        self._start_frame = self.render_frame()
        self._success = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        return self.render()

    @staticmethod
    def _restore(env, scene):
        env.set_ep_meta(scene["ep_meta"])
        env.reset()  # a hard reset loads the scene's fixtures; reset_from_xml_string alone does not
        env.reset_from_xml_string(env.edit_model_xml(scene["model"]))
        env.sim.reset()
        env.sim.set_state_from_flattened(scene["state"])
        env.sim.forward()
        env.update_state()
        return env._get_observations(force_update=True)

    def step(self, action):
        action = self._action(action)
        active = ~self.transition.done
        action[~active] = 0
        action = (self._low + (action.cpu() + 1) / 2 * (self._high - self._low)).numpy()
        obs, state, frames, success = self._obs.clone(), self.state.clone(), self._frames.clone(), self._success.clone()
        for lane in active.nonzero().flatten().tolist():
            env = self.envs[lane]
            observation, *_ = self._streams.call(lane, env.step, action[lane])
            lane_obs, lane_state, lane_frame = self._read([observation])
            obs[lane], state[lane], frames[lane] = lane_obs[0], lane_state[0], lane_frame[0]
            success[lane] = bool(self._streams.call(lane, env._check_success))
        self._obs, self.state, self._frames, self._success = obs, state, frames, success
        self.elapsed += active
        self.transition = Transition(
            (success & active).float(),
            self.transition.terminated | (success & active),
            self.transition.truncated | (~success & active & (self.elapsed >= self.max_episode_steps)),
            success.clone(),
        )
        return self.render()

    def render(self, state=None):
        if state is not None:
            raise ValueError("Arbitrary-state rendering is not supported by RoboCasa")
        return self._obs.clone()

    def render_frame(self, state=None):
        """Frames ``(B, 3, H, W)`` uint8 of the kitchen from RoboCasa's centered agent view, at ``render_size``."""
        if state is not None:
            raise ValueError("Arbitrary-state rendering is not supported by RoboCasa")
        return self._frames.clone()

    def render_goal_frame(self):
        """The scene at reset: RoboCasa states its goal in language, ``instruction``, not as an observation."""
        return self._start_frame.clone()

    @property
    def gt_label(self):
        return f"Task · {self.instruction[0]}" if self.instruction[0] else "Task"

    def close(self):
        for env in getattr(self, "envs", ()):
            env.close()
        self.envs = []
