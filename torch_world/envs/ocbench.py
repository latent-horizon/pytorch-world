import copy
import importlib.util
import os
import sys

import numpy as np
import torch

from torch_world.envs.benchmark import BenchmarkEnv, Transition, seed_from_generator
from torch_world.envs.lanes import suite_spec
from torch_world.envs.mujoco import close_renderers, import_mujoco


class OCBench(BenchmarkEnv):
    def __init__(
        self,
        name,
        *,
        batch_size=1,
        observation=None,
        size=None,
        camera="front",
        terminate_on_success=True,
        sim_backend="auto",
        quality="high",
        render_size=None,
        device=None,
        max_episode_steps=None,
        **env_kwargs,
    ):
        visual = name.startswith("visual-")
        if observation is None:
            observation = "rgb" if visual else "state"
        if observation not in ("state", "rgb"):
            raise ValueError("observation must be 'state' or 'rgb'")
        if visual and observation != "rgb":
            raise ValueError("Visual OCBench names observe RGB pixels; omit 'visual-' for state observations")
        if sim_backend not in ("auto", "cpu", "mjwarp"):
            raise ValueError("sim_backend must be 'auto', 'cpu', or 'mjwarp'")
        self.name = name
        self.observation = observation
        self.camera = camera
        self.terminate_on_success = terminate_on_success
        self.settings = {
            "name": name,
            "batch_size": batch_size,
            "observation": observation,
            "size": size,
            "camera": camera,
            "terminate_on_success": terminate_on_success,
            "sim_backend": sim_backend,
            "quality": quality,
            "render_size": render_size,
            "max_episode_steps": max_episode_steps,
            **env_kwargs,
        }
        self.spec = suite_spec(
            f"OCBench/{name}",
            "torch_world.envs.ocbench:OCBench",
            "ocbench",
            ("ocbench",),
            {"name": name},
            "https://github.com/seohongpark/ocbench",
        )
        super().__init__(batch_size, size, device, 1, quality=quality, render_size=render_size)
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            os.environ.setdefault("MUJOCO_GL", "egl")
        try:
            import ocbench
        except ImportError as exc:
            raise ImportError("Install OCBench with: pip install 'torch-world[ocbench]'") from exc
        try:
            family, spec_backend, _, default_steps = ocbench.parse_env_spec(name)
        except Exception as exc:
            raise ValueError(f"Unknown OCBench task {name!r}; tasks are named like 'block-single-task1-v0'") from exc
        if sim_backend == "mjwarp" and spec_backend == "cpu":
            raise ValueError("Pass a task name without '-cpu-' to use sim_backend='mjwarp'")
        use_mjwarp = sim_backend == "mjwarp" or (
            sim_backend == "auto"
            and spec_backend == "mjwarp"
            and torch.cuda.is_available()
            and importlib.util.find_spec("mujoco_warp") is not None
        )
        self.sim_backend = "mjwarp" if use_mjwarp else "cpu"
        self.has_success = family != "bowling"
        self.max_episode_steps = max_episode_steps or int(default_steps)
        target_name = name if use_mjwarp or spec_backend == "cpu" else self._cpu_name(name)
        self._upstream_render_size = self.size if observation == "rgb" else self.render_size
        make_kwargs = {
            "ob_type": "states",
            "width": self._upstream_render_size,
            "height": self._upstream_render_size,
            **env_kwargs,
        }
        if observation == "rgb":
            make_kwargs.setdefault("visualize_info", False)
        self.envs = []
        self._mjwarp_env = None
        self._renderers = {}
        self._start_cpu = None
        self._start_arrays = None
        self._start_frame = None
        try:
            if use_mjwarp:
                make_kwargs.pop("terminate_at_success", None)
                make_kwargs.pop("max_episode_steps", None)
                self._mjwarp_env = ocbench.make(target_name, nworld=batch_size, **make_kwargs)
                if hasattr(self._mjwarp_env, "_terminate_at_success"):
                    self._mjwarp_env._terminate_at_success = terminate_on_success
                if (
                    family == "bowling"
                    and hasattr(self._mjwarp_env, "cpu_env")
                    and hasattr(self._mjwarp_env.cpu_env, "_max_episode_steps")
                ):
                    self._mjwarp_env.cpu_env._max_episode_steps = self.max_episode_steps
                space = self._mjwarp_env.single_action_space
            else:
                make_kwargs["terminate_at_success"] = terminate_on_success
                if max_episode_steps is not None:
                    make_kwargs["max_episode_steps"] = max_episode_steps
                for _ in range(batch_size):
                    env = ocbench.make(target_name, **make_kwargs)
                    if family == "bowling" and hasattr(env.unwrapped, "_max_episode_steps"):
                        env.unwrapped._max_episode_steps = self.max_episode_steps
                    self.envs.append(env)
                space = self.envs[0].action_space
            self._low, self._high = (
                torch.as_tensor(np.asarray(bound, dtype=np.float32).ravel()) for bound in (space.low, space.high)
            )
            self.action_dim = self._low.numel()
            self.reset(seed=0)
            self.observation_shape = (3, self.size, self.size) if self.observation == "rgb" else (self.state_dim,)
        except Exception:
            self.close()
            raise

    @staticmethod
    def _cpu_name(name):
        visual = name.startswith("visual-")
        base = name.removeprefix("visual-")
        family, rest = base.split("-", 1)
        if rest.startswith("cpu-"):
            return name
        prefix = "visual-" if visual else ""
        return f"{prefix}{family}-cpu-{rest}"

    def _cpu_env(self, lane):
        if self._mjwarp_env is not None:
            return getattr(self._mjwarp_env, "cpu_env", None)
        return getattr(self.envs[lane], "unwrapped", self.envs[lane])

    def _physics(self, lane):
        cpu_env = self._cpu_env(lane)
        if cpu_env is None:
            return None
        try:
            return cpu_env.model, cpu_env.data
        except (AttributeError, ValueError):
            return None

    def _validate_camera(self):
        if not isinstance(self.camera, str):
            return
        physics = self._physics(0)
        if physics is None:
            return
        try:
            physics[0].camera(self.camera)
        except KeyError as exc:
            raise ValueError(f"Unknown OCBench camera {self.camera!r}") from exc

    def _snapshot_cpu(self, lane):
        mujoco = import_mujoco()
        model, data = self._physics(lane)
        kind = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(model, kind))
        mujoco.mj_getState(model, data, state, kind)
        return state, model.geom_rgba.copy()

    def _render_mujoco(self, key, model, data, resolution):
        mujoco = import_mujoco()
        if (key, resolution) not in self._renderers:
            buffer = model.vis.global_
            buffer.offwidth = max(buffer.offwidth, resolution)
            buffer.offheight = max(buffer.offheight, resolution)
            self._renderers[key, resolution] = mujoco.Renderer(model, resolution, resolution)
        renderer = self._renderers[key, resolution]
        renderer.update_scene(data, self.camera)
        return renderer.render()

    def _pixels(self, lanes, resolution):
        frames = []
        use_custom_renderer = resolution != self._upstream_render_size
        if self._mjwarp_env is not None:
            cpu_env = getattr(self._mjwarp_env, "cpu_env", None)
            physics = self._physics(0) if use_custom_renderer else None
            for lane in lanes:
                if physics is not None and hasattr(self._mjwarp_env, "copy_world_to_cpu_env"):
                    self._mjwarp_env.copy_world_to_cpu_env(lane)
                    frames.append(self._render_mujoco("mjwarp", cpu_env.model, cpu_env.data, resolution))
                elif hasattr(self._mjwarp_env, "render_world"):
                    frames.append(self._mjwarp_env.render_world(lane, camera=self.camera))
                else:
                    frames.append(self._mjwarp_env.render(camera=self.camera)[lane])
        else:
            for lane in lanes:
                cpu_env = self._cpu_env(lane)
                physics = self._physics(lane) if use_custom_renderer else None
                if physics is not None:
                    frames.append(self._render_mujoco(lane, physics[0], physics[1], resolution))
                else:
                    frames.append(cpu_env.render(camera=self.camera))
        return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()

    def reset(self, state=None, *, seed=None, generator=None):
        if state is not None:
            raise ValueError("OCBench observations are not simulator checkpoints; use reset(seed=...)")
        if seed is not None and generator is not None:
            raise ValueError("Pass seed or generator, not both")
        seed = seed_from_generator(generator) if seed is None else seed
        self._reset_episode()
        self._start_cpu = None
        self._start_arrays = None
        self._start_frame = None
        if self._mjwarp_env is not None:
            obs, _ = self._mjwarp_env.reset(seeds=[seed + lane for lane in range(self.batch_size)])
            self.state = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
            self._validate_camera()
            if hasattr(self._mjwarp_env, "_get_arrays") and hasattr(self._mjwarp_env, "copy_world_to_cpu_env"):
                self._start_arrays = {k: v.copy() for k, v in self._mjwarp_env._get_arrays().items()}
            else:
                self._start_frame = self.render_frame()
        else:
            states = []
            for lane, env in enumerate(self.envs):
                obs, _ = env.reset(seed=seed + lane)
                states.append(torch.as_tensor(np.asarray(obs, dtype=np.float32)))
            self.state = torch.stack(states).to(self.device)
            self._validate_camera()
            if self._physics(0) is not None:
                self._start_cpu = [self._snapshot_cpu(lane) for lane in range(self.batch_size)]
            else:
                self._start_frame = self.render_frame()
        self.state_dim = self.state.shape[-1]
        self._obs = (
            self.state.clone()
            if self.observation == "state"
            else self._pixels(range(self.batch_size), self.size).to(self.device).float() / 255
        )
        self._success = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        return self.render()

    def step(self, action):
        action = self._action(action)
        active = ~self.transition.done
        action[~active] = 0
        action_np = (self._low + (action.cpu() + 1) / 2 * (self._high - self._low)).numpy()
        state = self.state.clone()
        reward = torch.zeros(self.batch_size, device=self.device)
        terminated = torch.zeros_like(active)
        truncated = torch.zeros_like(active)
        success = self._success.clone()
        lanes = active.nonzero().flatten().tolist()
        if lanes:
            if self._mjwarp_env is not None:
                lane_state, r, term, trunc, info = self._mjwarp_env.step(action_np)
                lane_state = torch.as_tensor(np.asarray(lane_state, dtype=np.float32), device=self.device)
                r = torch.as_tensor(np.asarray(r, dtype=np.float32), device=self.device)
                term = torch.as_tensor(np.asarray(term, dtype=bool), device=self.device)
                trunc = torch.as_tensor(np.asarray(trunc, dtype=bool), device=self.device)
                lane_success = torch.as_tensor(
                    np.asarray(info.get("success", np.zeros(self.batch_size, dtype=bool)), dtype=bool),
                    device=self.device,
                )
                lane_failure = torch.as_tensor(
                    np.asarray(info.get("failure", np.zeros(self.batch_size, dtype=bool)), dtype=bool),
                    device=self.device,
                )
                lane_healthy = torch.as_tensor(
                    np.asarray(info.get("healthy", np.ones(self.batch_size, dtype=bool)), dtype=bool),
                    device=self.device,
                )
                if not self.has_success or not self.terminate_on_success:
                    term = lane_failure | ~lane_healthy
                state[active] = lane_state[active]
                reward[active] = r[active]
                terminated[active] = term[active]
                truncated[active] = trunc[active] & ~term[active]
                if self.has_success:
                    success[active] = lane_success[active]
            else:
                for lane in lanes:
                    obs, r, term, trunc, info = self.envs[lane].step(action_np[lane])
                    state[lane] = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
                    reward[lane] = float(r)
                    lane_failure = bool(info.get("failure", False))
                    lane_healthy = bool(info.get("healthy", True))
                    if not self.has_success or not self.terminate_on_success:
                        term = lane_failure or not lane_healthy
                    terminated[lane] = bool(term)
                    truncated[lane] = bool(trunc) and not bool(term)
                    if self.has_success:
                        success[lane] = bool(info.get("success", False))
            self.state, self._success = state, success
            if self.observation == "state":
                self._obs = self.state.clone()
            else:
                self._obs[lanes] = self._pixels(lanes, self.size).to(self.device).float() / 255
        self.elapsed += active
        timed_out = (self.elapsed >= self.max_episode_steps) & active & ~terminated
        self.transition = Transition(
            reward,
            self.transition.terminated | terminated,
            self.transition.truncated | truncated | timed_out,
            success.clone() if self.has_success else None,
        )
        return self.render()

    def render(self, state=None):
        if state is not None:
            raise ValueError("Arbitrary-state rendering is not supported by OCBench")
        return self._obs.clone()

    def render_frame(self, state=None):
        if state is not None:
            raise ValueError("Arbitrary-state rendering is not supported by OCBench")
        return self._pixels(range(self.batch_size), self.render_size)

    def render_goal_frame(self):
        if self._start_frame is not None:
            return self._start_frame.clone()
        if self._start_arrays is not None:
            cpu_env = self._mjwarp_env.cpu_env
            physics = self._physics(0) if self.render_size != self._upstream_render_size else None
            frames = []
            for lane in range(self.batch_size):
                self._mjwarp_env.copy_world_to_cpu_env(lane, arrays=self._start_arrays)
                if physics is not None:
                    frames.append(self._render_mujoco("mjwarp", cpu_env.model, cpu_env.data, self.render_size))
                else:
                    frames.append(cpu_env.render(camera=self.camera))
            return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()
        if self._start_cpu is None:
            raise RuntimeError("Call reset() before rendering the reset scene")
        mujoco = import_mujoco()
        saved = []
        try:
            for lane, (state, geom_rgba) in enumerate(self._start_cpu):
                model, data = self._physics(lane)
                saved.append((copy.copy(data), model.geom_rgba.copy()))
                mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
                model.geom_rgba[:] = geom_rgba
                mujoco.mj_forward(model, data)
            return self.render_frame()
        finally:
            for lane, (data, geom_rgba) in enumerate(saved):
                model, current = self._physics(lane)
                mujoco.mj_copyData(current, model, data)
                model.geom_rgba[:] = geom_rgba

    @property
    def gt_label(self):
        return "Task at reset"

    def close(self):
        close_renderers(getattr(self, "_renderers", {}))
        self._renderers = {}
        if getattr(self, "_mjwarp_env", None) is not None:
            self._mjwarp_env.close()
            self._mjwarp_env = None
        for env in getattr(self, "envs", ()):
            env.close()
        self.envs = []
