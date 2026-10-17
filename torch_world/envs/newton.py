"""Articulated swing-up benchmarks simulated by Newton's Featherstone solver."""

import math

import torch

from torch_world.envs.benchmark import BenchmarkEnv, Transition


class NewtonPendulum(BenchmarkEnv):
    """Torque-limited pendulum or elbow-actuated acrobot, with schematic RGB observations.

    Physics is Newton, not a handwritten integration approximation. Each batch lane
    has its own articulation. ``physics_device`` selects Warp CPU/CUDA separately
    from the torch observation ``device``. State contains joint angles and velocities.
    Success requires all links upright and nearly stationary; episodes do not end
    on transient success so stabilization can be evaluated.
    """

    def __init__(
        self,
        links=1,
        fully_actuated=True,
        *,
        batch_size=1,
        size=None,
        quality="high",
        render_size=None,
        device=None,
        max_episode_steps=300,
        physics_device="cpu",
        torque=4.0,
        substeps=10,
        dt=0.02,
    ):
        super().__init__(batch_size, size, device, max_episode_steps, quality=quality, render_size=render_size)
        if links not in (1, 2) or torque <= 0 or dt <= 0 or substeps < 1:
            raise ValueError("links must be 1 or 2; torque, dt and substeps must be positive")
        try:
            import newton
            import warp as wp
        except ImportError as exc:
            raise ImportError("Install Newton with: pip install 'torch-world[newton]'") from exc
        self.newton, self.wp = newton, wp
        self.links, self.fully_actuated = links, fully_actuated
        self.action_dim = links if fully_actuated else 1
        self.state_dim = 2 * links
        self.torque, self.substeps, self.dt = torque, substeps, dt
        builder = newton.ModelBuilder()
        parent, joints = -1, []
        for j in range(links):
            body = builder.add_link()
            builder.add_shape_box(body, hx=0.035, hy=0.035, hz=0.5, cfg=newton.ModelBuilder.ShapeConfig(density=200.0))
            joints.append(
                builder.add_joint_revolute(
                    parent,
                    body,
                    axis=(0, 1, 0),
                    parent_xform=wp.transform((0, 0, 0 if j == 0 else -0.5), wp.quat_identity()),
                    child_xform=wp.transform((0, 0, 0.5), wp.quat_identity()),
                    target_ke=0.0,
                    target_kd=0.0,
                    damping=0.02,
                    limit_lower=-1e6,
                    limit_upper=1e6,
                )
            )
            parent = body
        builder.add_articulation(joints)
        worlds = newton.ModelBuilder()
        for _ in range(batch_size):
            worlds.begin_world()
            worlds.add_builder(builder)
            worlds.end_world()
        self.model = worlds.finalize(device=physics_device)
        self.solver = newton.solvers.SolverFeatherstone(self.model)
        self.input, self.output = self.model.state(), self.model.state()
        self.control = self.model.control()
        self._torque_mask = torch.ones(links, device=self.device)
        if not fully_actuated:
            self._torque_mask[:-1] = 0
        self.state = torch.zeros(batch_size, self.state_dim, device=self.device)
        self.goal = self.state.clone()
        self.goal[:, 0] = math.pi

    def reset(self, state=None, generator=None, *, seed=None):
        if seed is not None and generator is not None:
            raise ValueError("Pass seed or generator, not both")
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        if state is None:
            state = (
                torch.rand(
                    self.batch_size, self.state_dim, generator=generator, device=getattr(generator, "device", "cpu")
                )
                - 0.5
            ) * 0.1
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state.shape != (self.batch_size, self.state_dim) or not torch.isfinite(state).all():
            raise ValueError("Invalid Newton state")
        self._reset_episode()
        self.state = state.clone()
        self._set_state(state)
        self.control.joint_f.zero_()
        return self.render()

    def _assign(self, destination, tensor):
        view = self.wp.to_torch(destination)
        view.copy_(tensor.detach().reshape(view.shape).to(view.device))

    def _set_state(self, state):
        self._assign(self.input.joint_q, state[:, : self.links])
        self._assign(self.input.joint_qd, state[:, self.links :])
        self.newton.eval_fk(self.model, self.input.joint_q, self.input.joint_qd, self.input)

    def step(self, action):
        action = self._action(action)
        active = ~self.transition.done
        forces = action * self._torque_mask * self.torque * active[:, None]
        self._assign(self.control.joint_f, forces)
        for _ in range(self.substeps):
            self.input.clear_forces()
            self.solver.step(self.input, self.output, self.control, None, self.dt / self.substeps)
            self.input, self.output = self.output, self.input
        rows = torch.cat(
            (
                self.wp.to_torch(self.input.joint_q).reshape(self.batch_size, self.links),
                self.wp.to_torch(self.input.joint_qd).reshape(self.batch_size, self.links),
            ),
            -1,
        ).to(self.device)
        self.state = torch.where(active[:, None], rows, self.state)
        self._set_state(self.state)
        self.elapsed += active
        reward = -self.distance(self.state, self.goal) - 0.001 * action.square().sum(-1)
        self.transition = Transition(
            reward * active,
            torch.zeros_like(active),
            self.elapsed >= self.max_episode_steps,
            self.success(self.state, self.goal),
        )
        return self.render()

    def distance(self, states, goal):
        goal = goal.view(len(goal), *([1] * (states.ndim - 2)), -1)
        angles = states[..., : self.links].cumsum(-1) - goal[..., : self.links].cumsum(-1)
        return (1 - angles.cos()).mean(-1) + 0.01 * (states[..., self.links :] - goal[..., self.links :]).square().mean(
            -1
        )

    def success(self, states, goal):
        return self.distance(states, goal) < 0.04

    def render(self, state=None):
        return self._render_at(state, self.size)

    def render_frame(self, state=None):
        return self._render_at(state, self.render_size)

    def render_goal_frame(self):
        return self.render_frame(self.goal)

    def _render_at(self, state, size):
        state = self.state if state is None else state.to(self.device)
        grid = torch.linspace(-self.links - 0.3, self.links + 0.3, size, device=self.device)
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        img = torch.full((len(state), 3, size, size), 0.95, device=self.device)
        angles = state[:, : self.links].cumsum(-1)
        x, y = torch.zeros(len(state), device=self.device), torch.zeros(len(state), device=self.device)
        for angle in angles.unbind(-1):
            nx, ny = x + angle.sin(), y + angle.cos()
            dx, dy = nx - x, ny - y
            t = ((xx - x[:, None, None]) * dx[:, None, None] + (yy - y[:, None, None]) * dy[:, None, None]).clamp(0, 1)
            dist = (xx - x[:, None, None] - t * dx[:, None, None]).square() + (
                yy - y[:, None, None] - t * dy[:, None, None]
            ).square()
            mask = (dist < 0.008)[:, None]
            color = torch.tensor([0.15, 0.35, 0.7], device=self.device)[None, :, None, None]
            img = torch.where(mask, color, img)
            x, y = nx, ny
        return img
