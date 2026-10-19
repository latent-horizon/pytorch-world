"""ManiSkill 3 tasks behind the batched benchmark protocol: one GPU-parallel environment, or one per lane on CPU."""

import torch

from torch_world.envs.benchmark import BenchmarkEnv, Transition, seed_from_generator
from torch_world.envs.lanes import suite_spec


class ManiSkill(BenchmarkEnv):
    """A `ManiSkill 3 <https://github.com/haosulab/ManiSkill>`_ task, by its registered ID.

    ``name`` is any ManiSkill task, such as ``"PickCube-v1"``, ``"StackCube-v1"`` or ``"PushT-v1"``. With CUDA, all
    lanes run in one GPU-parallel ManiSkill environment; otherwise each lane runs its own CPU environment, which is
    the only kind ManiSkill's CPU simulation supports. ``sim_backend="physx_cpu"`` forces the CPU lanes.

    Observations are ManiSkill's flattened state ``(B, D)``, or with ``observation="rgb"`` the ``camera`` sensor's
    images ``(B, 3, size, size)`` in [0, 1]; ``state`` is always the flattened agent and task state. Actions follow
    ``control_mode``, ManiSkill's default for the robot unless given, rescaled from [-1, 1] to its bounds.

    Rewards are ManiSkill's ``reward_mode``, normalized dense by default; ``transition.success`` is its ``success``
    flag, and episodes end on success or ``fail``, as ManiSkill terminates them, or at the task's time limit. With
    ``terminate_on_success=False`` only ``fail`` ends them early, and ``success`` is reported at every step. Frames
    come from ManiSkill's human render camera. Rendering needs a Vulkan driver, MoltenVK on macOS; on Linux,
    ``render_backend="none"`` runs state observations without one, and without frames. ``**env_kwargs`` go to
    ``gymnasium.make``, such as ``robot_uids`` or ``reward_mode``.
    """

    def __init__(
        self,
        name,
        *,
        batch_size=1,
        observation="state",
        size=None,
        camera="base_camera",
        control_mode=None,
        terminate_on_success=True,
        sim_backend="auto",
        quality="high",
        render_size=None,
        device=None,
        max_episode_steps=None,
        **env_kwargs,
    ):
        if observation not in ("state", "rgb"):
            raise ValueError("observation must be 'state' or 'rgb'")
        self.name, self.observation, self.camera = name, observation, camera
        self.terminate_on_success = terminate_on_success
        self.settings = {
            "name": name,
            "batch_size": batch_size,
            "observation": observation,
            "size": size,
            "camera": camera,
            "control_mode": control_mode,
            "terminate_on_success": terminate_on_success,
            "sim_backend": sim_backend,
            "quality": quality,
            "render_size": render_size,
            "max_episode_steps": max_episode_steps,
            **env_kwargs,
        }
        self.spec = suite_spec(
            f"ManiSkill/{name}",
            "torch_world.envs.maniskill:ManiSkill",
            "maniskill",
            ("maniskill",),
            {"name": name},
            "https://github.com/haosulab/ManiSkill",
        )
        # The episode limit comes from the upstream registration, read once the environments exist.
        super().__init__(batch_size, size, device, 1, quality=quality, render_size=render_size)
        try:
            import gymnasium as gym
            import mani_skill.envs  # noqa: F401, registers the tasks
            from mani_skill.utils.registration import REGISTERED_ENVS
        except ImportError as exc:
            raise ImportError("Install ManiSkill with: pip install 'torch-world[maniskill]'") from exc
        if name not in REGISTERED_ENVS:
            raise ValueError(f"Unknown ManiSkill task {name!r}; tasks are named like 'PickCube-v1'")
        self.renders = env_kwargs.get("render_backend", "gpu") not in ("none", None)
        if observation == "rgb" and not self.renders:
            raise ValueError("Camera observations need a render backend")
        gpu = sim_backend in ("auto", "gpu", "physx_cuda") and torch.cuda.is_available()
        kwargs = {
            "obs_mode": "rgb+state" if observation == "rgb" else "state",
            "render_mode": "rgb_array" if self.renders else None,
            "sensor_configs": {"width": self.size, "height": self.size},
            "human_render_camera_configs": {"width": self.render_size, "height": self.render_size},
            "sim_backend": "physx_cuda" if gpu else "physx_cpu",
            **({} if control_mode is None else {"control_mode": control_mode}),
            **({} if max_episode_steps is None else {"max_episode_steps": max_episode_steps}),
            **env_kwargs,
        }
        # Each group is an upstream environment and the lanes it simulates.
        self.groups = []
        try:
            if gpu:
                self.groups.append((gym.make(name, num_envs=batch_size, **kwargs), list(range(batch_size))))
            else:
                for lane in range(batch_size):
                    self.groups.append((gym.make(name, num_envs=1, **kwargs), [lane]))
        except Exception:
            self.close()
            raise
        env = self.groups[0][0]
        space = env.unwrapped.single_action_space
        self._low, self._high = (torch.as_tensor(b, dtype=torch.float32) for b in (space.low, space.high))
        self.action_dim = space.shape[0]
        self.max_episode_steps = max_episode_steps or env.spec.max_episode_steps
        self.reset(seed=0)

    def _read(self, obs):
        """Observations and state vectors of one group, on ``device``."""
        state = (obs["state"] if isinstance(obs, dict) else obs).float().to(self.device)
        if self.observation == "state":
            return state.clone(), state
        images = obs["sensor_data"][self.camera]["rgb"][..., :3].permute(0, 3, 1, 2)
        return images.float().to(self.device) / 255, state

    def reset(self, state=None, *, seed=None, generator=None):
        if state is not None:
            raise ValueError("ManiSkill observations are not simulator checkpoints; use reset(seed=...)")
        if seed is not None and generator is not None:
            raise ValueError("Pass seed or generator, not both")
        seed = seed_from_generator(generator) if seed is None else seed
        self._reset_episode()
        observations, states = [], []
        for env, lanes in self.groups:
            obs, _ = env.reset(seed=[seed + lane for lane in lanes])
            lane_obs, lane_state = self._read(obs)
            observations.append(lane_obs)
            states.append(lane_state)
        self._obs, self.state = torch.cat(observations), torch.cat(states)
        self.state_dim = self.state.shape[-1]
        self._success = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self._start_frame = self.render_frame() if self.renders else None
        return self.render()

    def step(self, action):
        action = self._action(action)
        active = ~self.transition.done
        action[~active] = 0
        action = self._low.to(self.device) + (action + 1) / 2 * (self._high - self._low).to(self.device)
        obs, state, success = self._obs.clone(), self.state.clone(), self._success.clone()
        reward = torch.zeros(self.batch_size, device=self.device)
        terminated, truncated = torch.zeros_like(active), torch.zeros_like(active)
        for env, lanes in self.groups:
            if not active[lanes].any():  # finished lanes are not stepped again
                continue
            upstream = env.unwrapped.device
            lane_obs, r, term, trunc, info = env.step(action[lanes].to(upstream))
            lane_obs, lane_state = self._read(lane_obs)
            # A GPU group steps every lane; the finished ones keep their last outcome, absorbing.
            keep = active[lanes]
            index = torch.tensor(lanes, device=self.device)[keep]
            obs[index], state[index] = lane_obs[keep], lane_state[keep]
            reward[index] = r.float().to(self.device)[keep]
            if not self.terminate_on_success:  # ManiSkill terminates on success or fail; keep only failures
                term = info["fail"] if "fail" in info else torch.zeros_like(term)
            terminated[index] = term.to(self.device)[keep]
            truncated[index] = trunc.to(self.device)[keep]
            if "success" in info:
                success[index] = info["success"].to(self.device)[keep]
        self._obs, self.state, self._success = obs, state, success
        self.elapsed += active
        self.transition = Transition(
            reward,
            self.transition.terminated | terminated,
            self.transition.truncated | truncated | ((self.elapsed >= self.max_episode_steps) & active),
            success.clone(),
        )
        return self.render()

    def render(self, state=None):
        """The current observation; states live in the simulator, so they cannot be rendered from tensors."""
        if state is not None:
            raise ValueError("Arbitrary-state rendering is not supported by ManiSkill")
        return self._obs.clone()

    def render_frame(self, state=None):
        """Frames ``(B, 3, render_size, render_size)`` as uint8 from ManiSkill's human render camera."""
        if state is not None:
            raise ValueError("Arbitrary-state rendering is not supported by ManiSkill")
        if not self.renders:
            raise RuntimeError("This ManiSkill environment was made with render_backend='none'")
        frames = [env.render()[..., :3].permute(0, 3, 1, 2).cpu() for env, _ in self.groups]
        return torch.cat(frames).contiguous()

    def render_goal_frame(self):
        """The scene at reset: ManiSkill defines goals in the simulator, and draws goal markers in its frames."""
        if self._start_frame is None:
            raise RuntimeError("This ManiSkill environment was made with render_backend='none'")
        return self._start_frame.clone()

    @property
    def gt_label(self):
        return "Task at reset"

    def close(self):
        for env, _ in getattr(self, "groups", ()):
            env.close()
        self.groups = []
