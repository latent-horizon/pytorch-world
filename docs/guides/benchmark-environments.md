# Benchmark experiments

`torch_world.envs` registers 33 tasks: 28 original task implementations (five planning puzzles, 18 MuJoCo scenes, two Newton control systems and three Isaac Lab tabletop tasks) and benchmark recipes for the five original environments. Every task runs through one experiment protocol: `run_experiment` trains a world model on random-action data and compares model-based planning with random actions. The [Environments](../environments/index.md) page lists every task by category; to add one, see [Bring your own environment](custom-environment.md).

Scene geometry, observations, reward and success logic live in this repository. Isaac Lab supplies the Franka asset and simulation primitives; no upstream task configurations or task wrappers are used.

## Discover and run

```python
from torch_world.envs import KeyDoor
from torch_world.benchmarks import run_experiment

world = KeyDoor(quality="high")
world.show_env()  # initial observation and GT; no training
result = run_experiment(world, output_dir="runs/benchmarks")
world.show_success()
world.close()
```

Every new benchmark task has a named class exported from `torch_world.envs`, including `Maze`, `Sokoban`, `PegInsertion`, `NewtonAcrobot`, and `FrankaCubeLift`. Configure the world once in its constructor. `world.show_env()` resets and previews its first batch lane alongside **GT**; `world.show_success()` resets it to display an animated, verified reference episode alongside GT. Both reuse the same instance and leave it open.

`run_experiment(world)` resets the world during data collection and evaluation and leaves it open afterward, including on errors. It records the constructor settings and versioned task ID. Call `world.close()` when finished, or use `with KeyDoor(quality="high") as world:` for automatic cleanup. Do not pass `env_kwargs` together with an instance; configure that instance in its constructor.

Classes, registry IDs and `EnvSpec` recipes remain supported for configuration files and benchmark sweeps. With `run_experiment(KeyDoor, env_kwargs={"quality": "high"})`, the runner creates and closes its own world. `get_spec(world)` and `get_spec(KeyDoor)` expose the same metadata.

Listing environments imports no optional simulator. Filter with `list_envs(category="manipulation")`, `backend=` or `tag=`; `CATEGORIES` lists the categories in display order. `make(id, **overrides)` applies constructor overrides to the registered defaults. All new environments support context managers and `close()`.

| Member | Contract |
|---|---|
| `reset(seed=..., generator=...)` | Return batched observations; provide a seed **or** generator, not both |
| `step(action)` | Finite actions `(B, action_dim)`, clipped to `[-1, 1]`; return observations |
| `render_target()` | GT success-condition observation at model resolution |
| `render_frame()` / `render_goal_frame()` | Independent presentation frame and GT; simulator cameras are available even with state observations |
| `render()` | Current observation; RGB only when a pixel observation is configured |
| `state` | Observation/state snapshot; **not** a universal physics checkpoint |
| `transition.reward` | Float tensor `(B,)`, task-defined rewards |
| `transition.terminated`, `.truncated`, `.done` | Boolean tensors `(B,)`; termination and time limits remain separate |
| `transition.success` | Boolean tensor `(B,)` or `None` when there is no success signal |
| `collect(T)` | `observations`, `next_observations`, `states`, `actions`, `rewards`, `terminated`, `truncated`, `valid`, all `(B,T,...)`; `pixels` aliases image observations |

Finished lanes are externally absorbing until reset. Their padding has `valid=False`, zero reward and zero collected actions. Every collected action has a corresponding next observation; the transition reaching a terminal state is valid. This differs from the legacy pixel worlds' frame-sequence collection convention. Filter or mask `valid` when training; never train across padding/reset boundaries.

MuJoCo success requires both position tolerance and low scored-coordinate velocities: 0.5 units/s by default, 0.05 units/s for the Robotiq placement tasks. Those tasks additionally check finger opening and tool clearance.

Puzzles, MuJoCo and Newton tasks support `reset(state)` and `render(state)` for state-goal evaluation. Isaac Lab observations cannot restore physics state; compare those with seeded resets and `evaluation.benchmark`, not `evaluation.goal_reaching`.

## Install a backend

```bash
pip install -e ".[benchmark,notebooks,test]"  # 18 original MuJoCo tasks, notebook tools
pip install -e ".[newton]"                   # Newton articulated control
```

The benchmark extra pins MuJoCo 3.6.0, the version used to verify the stored reference actions. LeWAM pins MuJoCo 3.15.0 for its separate matched study; install these two extras in separate virtual environments. The resolver declares them mutually exclusive. Reference replay is version-sensitive (BoxGate fails under 3.15), so do not combine their results across simulator versions. The `action` extra remains available for existing environments. The core package still depends only on torch.

Newton uses its real Featherstone articulated-body solver, with CPU as the portable default. Pass `physics_device="cuda:0"` to use Warp/CUDA. `device` separately selects the output tensor device. Its images are schematic views of the simulated links, not photorealistic camera images.

Isaac Lab is installed separately in its supported GPU environment. Our scenes target **Isaac Lab 3.x** and its Newton/MJWarp backend. Follow [Isaac Lab installation](https://isaac-sim.github.io/IsaacLab/develop/source/setup/installation/index.html).

```python
with make("FrankaObstacleReach-v0", batch_size=64, device="cuda:0") as env:
    observation = env.reset(seed=7)
```

The environment owns the runtime by default; use `launch=False` inside an existing runtime. `isaac_scenes.build_config` creates our table, barriers/objects, observations, actions and objectives from scratch. A custom `cfg` can extend these scenes. Default policy observations contain joint positions/velocities, task position and target error. Success is a task-position error below 4 cm. Terminal observations are captured before internal autoresets. The shipped Isaac notebooks use state vectors for training and real cameras for previews, filmstrips, and GIFs. Pass `observation="rgb"` to train from the built-in camera; no custom configuration or callbacks are needed. `camera="task"` frames the arm and tabletop, while `camera="overview"` shows a wider scene. Terminal state and camera snapshots are both captured before autoreset. Custom camera configurations still require compatible `success_fn` and `target_fn` callbacks.

MuJoCo shares its compiled model across lanes and executes substeps in native code. Newton builds all worlds into one solver. Isaac Lab uses GPU batches. Puzzle stepping and rendering are vectorized over lanes; remaining loops represent sequential updates, scene construction, or individual MuJoCo renderer calls.

## Visual quality

`quality="high"` is the default for all 28 new tasks. Model observations and notebook presentation frames have separate sizes:

| Preset | RGB model input | Presentation / GIF frame | Rendering |
|---|---:|---:|---|
| `low` | 32 × 32 | 160 × 160 | No physics shadows or multisampling |
| `medium` | 64 × 64 | 320 × 320 | Shadows, 2× MuJoCo multisampling |
| `high` | 128 × 128 | 512 × 512 | Shadows, 4× MuJoCo multisampling; 2× per-axis Franka supersampling |

```python
with FrankaCubeLift(quality="high", camera="task", observation="rgb") as world:
    world.show_env()
# Lower model cost without reducing preview/GIF clarity:
with Maze(quality="high", size=32, render_size=512) as world:
    world.show_env()
```

Puzzles use antialiased, cached sprites with distinct silhouettes and square cells. MuJoCo uses model cameras fitted to the start/GT geometry and contrasting task objects. Free-flight presentation cameras expand to include the current robot pose when it leaves those bounds; model observations retain their fixed view. Free-flight scenes use directional lights so vehicles remain illuminated as they gain altitude. Franka uses Isaac Lab's Newton Warp camera renderer with the articulated robot meshes. These settings improve visual detail; they do not change the physics timestep, solver accuracy, collision geometry, rewards, or success thresholds. Newton pendulum/acrobot views remain schematic.

Presentation frames are stored as `uint8`, independently of floating-point model observations.

## Swap the model and planner

`examples/envs/` contains one concise notebook per task, in one folder per category (`puzzles/`, `control/`, `navigation/`, `manipulation/`); the [Environments](../environments/index.md) page links each one. Each executed notebook links its measured run artifacts and episode visuals. All use the same library runner and plots. The [custom model notebook](../examples/envs/custom_model_three_envs.ipynb) defines a small residual world model and compares it with the imported LeWM baseline on Maze, KeyDoor, and Sokoban, with GT references, episode GIFs, and a campaign summary CSV.

```python
from torch_world.benchmarks import observation_encoder, run_experiment, show_results
from torch_world.envs import Maze
from torch_world.models import LeWM
from torch_world.planning import CEM

def model_factory(observation_shape, action_dim):
    return LeWM(
        observation_encoder(observation_shape), action_dim, embed_dim=32, history=1,
        depth=2, num_heads=4, head_dim=8, mlp_dim=64, proj_dim=64, dropout=0,
    )

def planner_factory(action_dim):
    return CEM(8, action_dim, num_samples=96, num_elites=12, iterations=3, bounds=(-1, 1))

world = Maze(quality="high")
result = run_experiment(
    world, model_factory=model_factory, planner_factory=planner_factory,
    seed=7, updates=400, output_dir="runs/benchmarks",
)
show_results(result)
world.close()
```

Replace `model_factory` to compare another `WorldModel`; the environment does not know about LeWM. The factory returns a trainable model supporting `forward(observations, actions)["loss"]`, `encode`, `predict`, `rollout`, and `cost`. This compact protocol uses two observations per training example and requires `history=1`. The default encoder accepts RGB or state vectors; the result records which was used. A user model can supply its own encoder. `model_config` records additional user-defined hyperparameters. `functools.partial` factories are supported; their bound arguments are recorded automatically. Saved planner settings reflect the effective objective used during evaluation.

Native puzzle data enumerates reachable nonterminal states and samples random actions. Protocol `one-step-world-model-v2` uses continuous uniform controls in `[-1, 1]`, including waits, decoded by the same function as online stepping. Earlier v1 archives used only full-strength cardinal moves; compare protocol versions separately. Training and holdout **starting states** are disjoint, although successor images may overlap. This is coverage-oriented offline data, not random-policy occupancy or an expert demonstration dataset. Other tasks use separate collection seeds. Padding is removed before training. Model factories receive the same data for a given task/seed.

Goal-renderable tasks plan against a goal embedding. Other tasks fit a reward head on frozen model embeddings, then plan the predicted rewards and report actual simulator returns. Sparse-reward data can contain no successes, so that head may be uninformative; inspect measured returns rather than assuming a working controller. Latent MSE values are comparable against persistence **within** a model, not across differently scaled latent spaces.

The default 400 updates, four evaluation seeds, and 60 steps are a pipeline-validation budget. Fixed puzzles repeat a single initial state; confidence intervals reflect policy randomness on that level, not generalization to new levels. `show_results` uses `plots.curves`, `plots.bars`, `plots.images`, and `plots.episodes`; all 28 built-in tasks have RGB presentation frames and GIFs, including state-trained Franka tasks. User environments without a visual renderer retain vector trajectories. GIF subsampling always retains the final observation and labels frames with their original episode steps. Tasks without a success signal omit success metrics instead of reporting failure.

## Saved runs

Set `output_dir` to save each run independently:

```text
runs/benchmarks/<environment>/<model>/<UTC-time>-<unique-id>/
    config.json       # factories, environment settings, optimizer, planner, seeds, budgets
    metrics.json      # prediction metrics and per-policy/per-seed outcomes
    training.csv      # update, loss
    episodes.pt       # first evaluation seed per policy, tensor-only snapshots
    target.pt         # GT reference, independent of achieved observations
    <policy>.gif      # RGB episode with a persistent GT panel
    manifest.json     # schema version, package versions, file checksums; written last
```

Runs never overwrite one another. An interrupted run has no manifest and is excluded from summaries. `summarize_runs` verifies file hashes before reading results. Load episode tensors with `torch.load(path, weights_only=True)`. Factories and architecture are recorded; keep custom factory source code and checkpoints under your own version control when exact model reconstruction is required. The current validation runner saves observations and scores, not trained model checkpoints.

A later benchmark is a loop over tasks, models, and seeds using the same runner:

```python
from torch_world.benchmarks import run_experiment, write_summary

for task in ("Maze-v0", "KeyDoor-v0", "Sokoban-v0"):
    for seed in (7, 8, 9):
        run_experiment(task, seed=seed, output_dir="runs/benchmarks")

rows = write_summary("runs/benchmarks", "runs/benchmark-summary.csv")
```

Use `load_run(run_directory)` to verify and reload a saved run, then `show_results(result)` to redisplay its tables, GT and episodes without training again.

The CSV has one row per completed run/policy. Keep environments, observation modalities, training budgets and protocols separate when comparing; do not average raw returns across unrelated tasks.

## GT and reference episodes

Every result filmstrip and GIF includes a reference labeled **GT**. Franka reach uses a visible green marker labeled **GT · green target**. Push/lift GT places the object at its target and restores the simulator immediately afterward; the unchanged arm is not a solved manipulation demonstration. `env.render_goal_frame()` returns the visual reference without stepping physics, while `render_target()` retains the model-observation interface. It represents the success condition, not a model prediction.

```python
from torch_world.envs import KeyDoor, PegInsertion

with KeyDoor() as world:
    world.show_success()  # verified search-based reference episode + GT

with PegInsertion() as world:
    world.show_success(path="runs/references/peg_insertion.gif")  # verified episode GIF + GT
    # world.show_success(policy=my_controller)  # replay and verify a controller
```

`show_success()` always displays an episode GIF. It resets the world, executes real actions, and checks the simulator's success flag before displaying or saving anything. GT remains visible throughout. A failed or unavailable reference raises `ValueError`; the function never substitutes a static target illustration.

Puzzles use exact search. MuJoCo and Newton tasks replay recorded reference actions, verified again on each invocation; differential parking uses feedback to correct heading drift. Franka tasks use feedback controllers with simulator kinematics and contact physics. These privileged references never enter the LeWM benchmark scores. The Franka arm maps normalized actions to the robot asset's joint limits, so reaching and grasping targets are accessible.

**Default coverage:** 27 of the 28 tasks have verified reference episodes. DoubleCartpole still requires a supplied controller; its notebook marks the reference as pending. Recorded plans target the default physics settings and seed 7. Modified settings are still checked for success on every replay.

Reference episodes can be longer than the model evaluation budget. Parking, reaction-wheel swing-up, and acrobot swing-up need additional time. The helper temporarily extends the reference episode limit, restores the caller's limit afterward, and samples long videos while preserving actual simulator step numbers and the terminal frame. `path=` optionally saves the GIF; experiment budgets and scores remain separate.

A supplied `policy` receives batched observations and returns normalized actions. Contributors can implement `reference_policy()` to return a fresh controller after reset. A controller may expose `max_steps` to set its demonstration budget. Implement RGB `render_frame()` and `render_goal_frame()` even when model observations are state vectors. The controller may read privileged simulator state to illustrate the task, but it must succeed through ordinary `step()` calls.

Refresh notebook demonstrations without retraining their models:

```bash
python examples/envs/render_references.py Maze-v0
```

This writes the GIF and provenance under `runs/references/<environment>/<timestamp>/` and updates only the notebook's reference output. Baseline results are preserved.

## Original environments

The registry also includes five benchmark recipes for the original environments: `PointMass-v0`, `Macaw-v0`, `PushT-v0`, `RobotCube-v0` (UR10e) and `RobotCubeSO101-v0`. Their notebooks use the same imported LeWM/CEM baseline, replaceable factories, result tables, GT panels and episode GIFs. Use the classes directly:

```python
from torch_world.envs import Macaw, PointMass, RobotCube
from torch_world.benchmarks import run_experiment, show_results

world = Macaw(size=128, quality="high")  # or PointMass(...), RobotCube(...)
world.show_env()
result = run_experiment(world, output_dir="runs/benchmarks")
world.show_success()
show_results(result)
world.close()
```

These recipes use fixed GT throughout an experiment and random-action training data. PointMass, Macaw and RobotCube keep seeded random starts. PushT uses a fixed symmetric translation task, not the full random-pose distribution. RobotCube benchmark success additionally requires released fingers and tool withdrawal; the original class's scoring is unchanged. Its notebook camera shows the whole arm, while model inputs retain the original task camera. Set `presentation_camera="task"` to use that view in GIFs too. Set constructor options directly, for example `RobotCube(robot="so101", camera="wrist", size=128)`.

`PushT()` supports the same workflow. All four classes preserve their original positional arguments, simulation methods and frame-sequence `collect()` API. The runner borrows a cached episode view of the same world, adding fixed goals, aligned transitions and episode limits; it leaves the world open afterward. Use `world.as_benchmark()` to access that protocol directly. `observation="state"` configures the benchmark view; original simulation methods still return pixels. `quality` and `render_size` control presentation; `size` retains its original observation default, so the notebooks explicitly select 128 pixels. String recipes through `make()` remain supported.

## Hardware models

Macaw's presentation frames preserve the animated wing phase; its GT is the gliding pose. A single frame does not reveal phase, so the shared `history=1` experiment has a known limitation here. The original scripted play-data collectors remain available independently of this random-action benchmark protocol.

Most MuJoCo tasks use primitive geometry for controlled experiments, including explicitly constrained planar systems. `PickPlace` and `BlockStack` use the articulated Robotiq 2F-85 hardware model on a Cartesian stage. Its meshes, inertias, four-bar linkages, tendon drive and contact pads are packaged locally, with the upstream BSD license and pinned provenance in `torch_world/envs/assets/robotiq_2f85/`. The surrounding tasks are our own implementation. Isaac Lab tasks use the articulated Franka asset. These are validation environments, not claims of sim-to-real fidelity.
