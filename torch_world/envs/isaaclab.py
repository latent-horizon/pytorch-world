"""Original Isaac Lab scenes with batched, boundary-safe experiment integration."""

import torch

from torch_world.envs.benchmark import BenchmarkEnv, Transition, seed_from_generator


class IsaacLabEnv(BenchmarkEnv):
    """Run our Franka scenes in Isaac Lab 3.x with Newton physics.

    Robot assets and engine primitives come from Isaac Lab. Scene layout,
    observations, rewards and objectives are defined in ``isaac_scenes``.
    Runtime lifetime is owned here unless ``launch=False``.
    """

    def __init__(
        self,
        task,
        *,
        cfg=None,
        batch_size=1,
        size=None,
        quality="high",
        render_size=None,
        camera="task",
        observation="state",
        device="cuda:0",
        max_episode_steps=500,
        observation_key="policy",
        launch=True,
        launcher_args=None,
        success_fn=None,
        target_fn=None,
    ):
        try:
            from isaaclab.app import launch_simulation
            from isaaclab.envs import ManagerBasedRLEnv

            from torch_world.envs.isaac_scenes import build_config
        except ImportError as exc:
            raise ImportError(
                "IsaacLabEnv requires Isaac Lab 3.x in a supported GPU environment; "
                "see docs/guides/benchmark-environments.md"
            ) from exc
        super().__init__(batch_size, size, device, max_episode_steps, quality=quality, render_size=render_size)
        if observation not in ("state", "rgb"):
            raise ValueError("observation must be state or rgb")
        self.observation, self.camera = observation, camera
        self._builtin_scene = cfg is None
        self._frame = None
        self._goal_frame = None
        self.task = task
        self.observation_key = "camera" if observation == "rgb" and cfg is None else observation_key
        self.success_fn = success_fn
        self.target_fn = target_fn
        self.runtime, self.env = None, None
        if cfg is None:
            cfg = build_config(
                task,
                batch_size,
                max_episode_steps,
                render_size=self.render_size * (2 if quality == "high" else 1),
                shadows=self.render_quality.shadows,
                camera=camera,
            )
        cfg.scene.num_envs = batch_size
        cfg.sim.device = str(self.device)
        if not hasattr(cfg, "compute_final_obs"):
            raise ValueError("Isaac Lab 3.x with compute_final_obs is required to preserve terminal observations")
        cfg.compute_final_obs = True
        try:
            if launch:
                self.runtime = launch_simulation(cfg, launcher_args or {"headless": True})
                self.runtime.__enter__()
            self.env = ManagerBasedRLEnv(cfg=cfg)
            self.action_dim = self.env.unwrapped.action_space.shape[-1]
            self.state = None
        except Exception:
            self.close()
            raise

    def reference_policy(self):
        if not self._builtin_scene:
            return super().reference_policy()
        from torch_world.envs.isaac_reference import FrankaReference

        return FrankaReference(self)

    def _observation(self, obs):
        if isinstance(obs, dict):
            obs = obs[self.observation_key]
        if not isinstance(obs, torch.Tensor):
            raise TypeError("Select a tensor observation group with observation_key")
        obs = obs.to(self.device).clone()
        if obs.ndim == 4 and obs.shape[-1] in (3, 4):
            obs = obs[..., :3].permute(0, 3, 1, 2)
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255
        else:
            obs = obs.float()
        if obs.ndim == 4:
            if not self._builtin_scene and (self.success_fn is None or self.target_fn is None):
                raise ValueError(
                    "Camera observations require both success_fn and target_fn for verified outcomes and GT"
                )
            obs = torch.nn.functional.interpolate(obs, (self.size, self.size), mode="bilinear", align_corners=False)
        return obs

    def reset(self, state=None, generator=None, *, seed=None):
        if state is not None:
            raise ValueError("Use reset(seed=...) for Isaac Lab; observation vectors are not physics checkpoints")
        if seed is not None and generator is not None:
            raise ValueError("Pass seed or generator, not both")
        self._reset_episode()
        self._goal_frame = None
        obs, _ = self.env.reset(seed=seed_from_generator(generator) if seed is None else seed)
        self._obs = self._observation(obs)
        self.state = obs["policy"].clone() if self._builtin_scene else self._obs.flatten(1).clone()
        self._frame = self._camera_frame(obs)
        self.state_dim = self.state.shape[-1]
        return self.render()

    def step(self, action):
        action = self._action(action)
        active = ~self.transition.done
        action[~active] = 0
        obs, reward, terminated, truncated, info = self.env.step(action)
        current = self._observation(obs)
        state = obs["policy"].clone() if getattr(self, "_builtin_scene", False) else current.flatten(1).clone()
        frame = self._camera_frame(obs)
        done = terminated.bool() | truncated.bool()
        if (done & active).any():
            if "final_obs" not in info:
                raise RuntimeError("Isaac Lab did not supply final_obs at autoreset; refusing corrupted trajectories")
            terminal = self._observation(info["final_obs"])
            current[done] = terminal[done]
            terminal_state = (
                info["final_obs"]["policy"] if getattr(self, "_builtin_scene", False) else terminal.flatten(1)
            )
            state[done] = terminal_state[done]
            if frame is not None:
                frame[done] = self._camera_frame(info["final_obs"])[done]
        current[~active] = self._obs[~active]
        self._obs = current
        previous = getattr(self, "state", None)
        if previous is not None:
            state[~active] = previous[~active]
        self.state = state
        if frame is not None:
            frame[~active] = self._frame[~active]
            self._frame = frame
        self.elapsed += active
        if self.success_fn is not None:
            success = torch.as_tensor(self.success_fn(self.env.unwrapped, info), device=self.device, dtype=torch.bool)
            if success.shape != active.shape:
                raise ValueError("success_fn must return one boolean per batch lane")
        elif getattr(self, "_builtin_scene", False):
            success = self.state[:, -3:].norm(dim=-1) < 0.04
        else:
            success = None
        self.transition = Transition(
            reward.to(self.device).clone() * active,
            self.transition.terminated | (terminated.bool() & active),
            self.transition.truncated
            | (truncated.bool() & active)
            | ((self.elapsed >= self.max_episode_steps) & active),
            success,
            info,
        )
        return self.render()

    @property
    def gt_label(self):
        return "GT · green target" if self.task == "reach" else "GT · object target"

    def _camera_frame(self, observations):
        if not isinstance(observations, dict) or "camera" not in observations:
            return None
        frame = observations["camera"][..., :3].permute(0, 3, 1, 2)
        if frame.shape[-2:] != (self.render_size, self.render_size):
            frame = (
                torch.nn.functional.interpolate(frame.float(), (self.render_size, self.render_size), mode="area")
                .round()
                .to(torch.uint8)
            )
        return frame.clone()

    def render_frame(self, state=None):
        if state is not None:
            raise ValueError("Arbitrary-state camera rendering is not supported by IsaacLabEnv")
        return self._frame.clone() if self._frame is not None else self.render()

    def render_goal_frame(self):
        """Camera GT: green reach marker, or object placed at its required position.

        The arm is unchanged: these are success-condition illustrations, not
        demonstrations of a successful manipulation policy. No physics step runs.
        """
        if not self._builtin_scene:
            return self.render_target()
        if self._goal_frame is None:
            if self.task == "reach":
                self._goal_frame = self.render_frame()
            else:
                from torch_world.envs.isaac_scenes import TARGETS

                obj = self.env.scene["object"]
                original = obj.data.root_state_w.clone()
                target = original.clone()
                target[:, :3] = self.env.scene.env_origins + torch.tensor(TARGETS[self.task], device=self.device)
                target[:, 7:] = 0
                try:
                    obj.write_root_state_to_sim(target)
                    self.env.sim.forward()
                    self.env.sim.render()
                    self.env.scene["camera"].update(0, force_recompute=True)
                    self._goal_frame = self._camera_frame({"camera": self.env.scene["camera"].data.output["rgb"]})
                finally:
                    obj.write_root_state_to_sim(original)
                    self.env.sim.forward()
                    self.env.sim.render()
                    self.env.scene["camera"].update(0, force_recompute=True)
        return self._goal_frame.clone()

    def render(self, state=None):
        if state is not None:
            raise ValueError("Arbitrary-state rendering is not supported by IsaacLabEnv")
        return self._obs.clone()

    def render_target(self):
        """GT specifies task position and zero target error; robot joints are unspecified."""
        from torch_world.envs.isaac_scenes import TARGETS

        if self.target_fn is not None:
            return self._observation(self.target_fn(self.env.unwrapped))
        if self.observation == "rgb":
            return self._observation({"camera": self.render_goal_frame().permute(0, 2, 3, 1)})
        target = torch.full_like(self._obs, float("nan"))
        target[:, -6:-3] = torch.tensor(TARGETS[self.task], device=self.device)
        target[:, -3:] = 0
        return target

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
        if self.runtime is not None:
            self.runtime.__exit__(None, None, None)
            self.runtime = None
