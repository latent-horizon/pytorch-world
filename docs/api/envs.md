# torch_world.envs

::: torch_world.envs.PointMass

::: torch_world.envs.Macaw

::: torch_world.envs.robot_cube.RobotCube

::: torch_world.envs.run_policy


## Registered benchmark tasks

::: torch_world.envs.EnvSpec

::: torch_world.envs.list_envs

::: torch_world.envs.make

::: torch_world.envs.register_env

::: torch_world.envs.register_task

::: torch_world.envs.BenchmarkEnv

::: torch_world.envs.Transition

::: torch_world.benchmarks.run_experiment

::: torch_world.benchmarks.save_run

::: torch_world.benchmarks.summarize_runs

::: torch_world.benchmarks.write_summary

::: torch_world.envs.Scene

::: torch_world.benchmarks.show_success

## Named environment classes

Import tasks directly, for example `from torch_world.envs import KeyDoor, PegInsertion`.
Create `world = KeyDoor(**settings)`, preview it with `world.show_env()`, and run
`run_experiment(world)`. The world remains open for `world.show_success()` or further
experiments. Call `world.close()` when finished, or use a context manager.

Passing a class or recipe to `run_experiment(KeyDoor, env_kwargs=settings)` still
creates an environment owned and closed by the runner. Use `get_spec(world)` or
`get_spec(KeyDoor)` for versioned metadata.

## Rendering quality

All 33 registered recipes accept `quality="low" | "medium" | "high"` (default `high`).
`size` overrides RGB model resolution; `render_size` overrides presentation resolution.
`render_frame()` and `render_goal_frame()` return batched camera/presentation frames independently
of the policy observation. `render_settings` exposes effective resolutions for saved experiments.
MuJoCo and Franka accept `camera="task" | "overview"`; Franka also accepts `observation="rgb" | "state"`.

::: torch_world.envs.RenderQuality

## Original-environment benchmark recipes

Use `PointMass()`, `Macaw()`, `PushT()` or `RobotCube()` directly with `run_experiment(world, ...)`, `world.show_env()` and `world.show_success()`. Select the SO-101 with `RobotCube(robot="so101")`. Success displays are verified episode GIFs with GT.

The original constructors and frame-sequence `collect()` interfaces remain supported. `world.as_benchmark()` exposes the shared episode view, with fixed goals, seeded resets, episode limits and aligned transition collection. Experiments use this view automatically and share the original simulator state. The `observation` constructor option selects the benchmark observation type; original reset/step/render methods continue returning pixels. `size` retains each original class's default; use `size=128` to match the notebooks. Presentation defaults to high quality at 512 pixels. RobotCube previews default to a full-arm overview; `presentation_camera="task"` selects its original camera. Registered string recipes through `make()` remain available.

::: torch_world.envs.legacy.LegacyEnvironment

::: torch_world.envs.legacy.LegacyTask
