# Environments

`torch_world.envs` registers 33 tasks for training and evaluating world models, in four categories. Every task is batched, exposes the same `reset` / `step` / `collect` interface, and has a notebook that trains LeWM and plans with CEM.

The **Success** column shows a verified reference episode from `world.show_success()`: real simulator steps that end in the task's success condition. These references are privileged controllers or search, not model results.

| Category | `category=` | Tasks | Backends |
|---|---|---:|---|
| [Planning puzzles](#planning-puzzles) | `"puzzles"` | 5 | PyTorch |
| [Control](#control) | `"control"` | 6 | MuJoCo, Newton |
| [Navigation and locomotion](#navigation-and-locomotion) | `"navigation"` | 6 | PyTorch, MuJoCo |
| [Manipulation](#manipulation) | `"manipulation"` | 16 | MuJoCo, Isaac Lab |

## Use an environment

```python
from torch_world.envs import KeyDoor, list_envs
from torch_world.benchmarks import run_experiment, show_results

print([spec.id for spec in list_envs(category="puzzles")])

with KeyDoor(quality="high") as world:
    world.show_env()      # first observation next to GT
    result = run_experiment(world, output_dir="runs/benchmarks")
    world.show_success()  # verified reference episode
    show_results(result)
```

Listing never imports an optional simulator. Puzzles need only PyTorch. MuJoCo tasks need `pip install -e ".[benchmark]"`, Newton tasks `pip install -e ".[newton]"`, and the Franka tasks an Isaac Lab 3.x GPU runtime; see [Environment benchmarks](../guides/benchmark-environments.md#install-the-relevant-backend). Tasks marked *original* also keep their earlier interfaces; see [Pixel worlds](../guides/environments.md).

## Planning puzzles

Discrete grid worlds written in pure PyTorch, with no simulator to install. They test long-horizon planning: dead ends, prerequisites, ordered subgoals, momentum and irreversible moves.

| Success | Environment | Backend | Task | Notebook |
|:---:|---|---|---|---|
| ![Checkpoints-v0 verified success](../assets/envs/success/checkpoints_v0.gif){ width="120" loading=lazy } | `Checkpoints-v0` | PyTorch | Visit two checkpoints in order before reaching the exit | [notebook](../examples/envs/puzzles/checkpoints_v0.ipynb) |
| ![IceMaze-v0 verified success](../assets/envs/success/icemaze_v0.gif){ width="120" loading=lazy } | `IceMaze-v0` | PyTorch | Slide until blocked; plan stopping positions | [notebook](../examples/envs/puzzles/icemaze_v0.ipynb) |
| ![KeyDoor-v0 verified success](../assets/envs/success/keydoor_v0.gif){ width="120" loading=lazy } | `KeyDoor-v0` | PyTorch | Collect a key before crossing a locked door | [notebook](../examples/envs/puzzles/keydoor_v0.ipynb) |
| ![Maze-v0 verified success](../assets/envs/success/maze_v0.gif){ width="120" loading=lazy } | `Maze-v0` | PyTorch | Route around walls and dead ends | [notebook](../examples/envs/puzzles/maze_v0.ipynb) |
| ![Sokoban-v0 verified success](../assets/envs/success/sokoban_v0.gif){ width="120" loading=lazy } | `Sokoban-v0` | PyTorch | Push a box to a target without trapping it | [notebook](../examples/envs/puzzles/sokoban_v0.ipynb) |

## Control

Underactuated and balancing systems: swing-up, stabilization and tilt control under torque limits, simulated with MuJoCo or Newton.

| Success | Environment | Backend | Task | Notebook |
|:---:|---|---|---|---|
| ![BallPlate-v0 verified success](../assets/envs/success/ballplate_v0.gif){ width="120" loading=lazy } | `BallPlate-v0` | MuJoCo | Tilt a plate to position a constrained ball | [notebook](../examples/envs/control/ballplate_v0.ipynb) |
| ![CartpoleSwingup-v0 verified success](../assets/envs/success/cartpoleswingup_v0.gif){ width="120" loading=lazy } | `CartpoleSwingup-v0` | MuJoCo | Torque-limited cart-pole swing-up | [notebook](../examples/envs/control/cartpoleswingup_v0.ipynb) |
| *reference pending* | `DoubleCartpole-v0` | MuJoCo | Underactuated double-pole stabilization | [notebook](../examples/envs/control/doublecartpole_v0.ipynb) |
| ![ReactionWheel-v0 verified success](../assets/envs/success/reactionwheel_v0.gif){ width="120" loading=lazy } | `ReactionWheel-v0` | MuJoCo | Reaction-wheel pendulum balance | [notebook](../examples/envs/control/reactionwheel_v0.ipynb) |
| ![NewtonAcrobot-v0 verified success](../assets/envs/success/newtonacrobot_v0.gif){ width="120" loading=lazy } | `NewtonAcrobot-v0` | Newton | Underactuated two-link swing-up with Newton physics | [notebook](../examples/envs/control/newtonacrobot_v0.ipynb) |
| ![NewtonPendulum-v0 verified success](../assets/envs/success/newtonpendulum_v0.gif){ width="120" loading=lazy } | `NewtonPendulum-v0` | Newton | Torque-limited swing-up with Newton physics | [notebook](../examples/envs/control/newtonpendulum_v0.ipynb) |

## Navigation and locomotion

Move a body to a goal: a point mass, an animated macaw, a planar quadrotor with and without a suspended payload, a differential-drive robot and a hopping leg.

| Success | Environment | Backend | Task | Notebook |
|:---:|---|---|---|---|
| ![Macaw-v0 verified success](../assets/envs/success/macaw_v0.gif){ width="120" loading=lazy } | `Macaw-v0` | PyTorch | Fly the animated macaw to a target position *(original)* | [notebook](../examples/envs/navigation/macaw_v0.ipynb) |
| ![PointMass-v0 verified success](../assets/envs/success/pointmass_v0.gif){ width="120" loading=lazy } | `PointMass-v0` | PyTorch | Move a point mass to a target in the unit square *(original)* | [notebook](../examples/envs/navigation/pointmass_v0.ipynb) |
| ![DifferentialParking-v0 verified success](../assets/envs/success/differentialparking_v0.gif){ width="120" loading=lazy } | `DifferentialParking-v0` | MuJoCo | Park a differential-drive robot between barriers | [notebook](../examples/envs/navigation/differentialparking_v0.ipynb) |
| ![HurdleHopper-v0 verified success](../assets/envs/success/hurdlehopper_v0.gif){ width="120" loading=lazy } | `HurdleHopper-v0` | MuJoCo | Hop a single articulated leg across a hurdle | [notebook](../examples/envs/navigation/hurdlehopper_v0.ipynb) |
| ![PayloadTransport-v0 verified success](../assets/envs/success/payloadtransport_v0.gif){ width="120" loading=lazy } | `PayloadTransport-v0` | MuJoCo | Transport a suspended payload without swing | [notebook](../examples/envs/navigation/payloadtransport_v0.ipynb) |
| ![PlanarQuadrotor-v0 verified success](../assets/envs/success/planarquadrotor_v0.gif){ width="120" loading=lazy } | `PlanarQuadrotor-v0` | MuJoCo | Differential-thrust planar flight | [notebook](../examples/envs/navigation/planarquadrotor_v0.ipynb) |

## Manipulation

Arms and grippers that act through contact: reaching, pushing, sliding, insertion, opening doors and drawers, grasping and stacking. MuJoCo scenes use primitive arms, the Robotiq 2F-85, the UR10e and the SO-101; Isaac Lab tasks use the Franka.

| Success | Environment | Backend | Task | Notebook |
|:---:|---|---|---|---|
| ![BlockStack-v0 verified success](../assets/envs/success/blockstack_v0.gif){ width="120" loading=lazy } | `BlockStack-v0` | MuJoCo | Stack a free cube on a raised block with a Robotiq 2F-85 | [notebook](../examples/envs/manipulation/blockstack_v0.ipynb) |
| ![BoxGate-v0 verified success](../assets/envs/success/boxgate_v0.gif){ width="120" loading=lazy } | `BoxGate-v0` | MuJoCo | Push a box through a narrow gate | [notebook](../examples/envs/manipulation/boxgate_v0.ipynb) |
| ![DoorOpen-v0 verified success](../assets/envs/success/dooropen_v0.gif){ width="120" loading=lazy } | `DoorOpen-v0` | MuJoCo | Push a hinged door open | [notebook](../examples/envs/manipulation/dooropen_v0.ipynb) |
| ![DrawerOpen-v0 verified success](../assets/envs/success/draweropen_v0.gif){ width="120" loading=lazy } | `DrawerOpen-v0` | MuJoCo | Contact a handle and pull a sliding drawer | [notebook](../examples/envs/manipulation/draweropen_v0.ipynb) |
| ![ObstacleArm-v0 verified success](../assets/envs/success/obstaclearm_v0.gif){ width="120" loading=lazy } | `ObstacleArm-v0` | MuJoCo | Two-link arm positioning around an obstacle | [notebook](../examples/envs/manipulation/obstaclearm_v0.ipynb) |
| ![PegInsertion-v0 verified success](../assets/envs/success/peginsertion_v0.gif){ width="120" loading=lazy } | `PegInsertion-v0` | MuJoCo | Insert a Cartesian peg into a clearance fixture | [notebook](../examples/envs/manipulation/peginsertion_v0.ipynb) |
| ![PickPlace-v0 verified success](../assets/envs/success/pickplace_v0.gif){ width="120" loading=lazy } | `PickPlace-v0` | MuJoCo | Grasp and relocate a free cube with a Robotiq 2F-85 | [notebook](../examples/envs/manipulation/pickplace_v0.ipynb) |
| ![PuckPush-v0 verified success](../assets/envs/success/puckpush_v0.gif){ width="120" loading=lazy } | `PuckPush-v0` | MuJoCo | Contact-only planar object pushing | [notebook](../examples/envs/manipulation/puckpush_v0.ipynb) |
| ![PuckSlide-v0 verified success](../assets/envs/success/puckslide_v0.gif){ width="120" loading=lazy } | `PuckSlide-v0` | MuJoCo | Slide an object beyond the tool workspace by releasing momentum | [notebook](../examples/envs/manipulation/puckslide_v0.ipynb) |
| ![PushT-v0 verified success](../assets/envs/success/pusht_v0.gif){ width="120" loading=lazy } | `PushT-v0` | MuJoCo | Push a T-shaped block to a fixed target pose *(original)* | [notebook](../examples/envs/manipulation/pusht_v0.ipynb) |
| ![RobotCube-v0 verified success](../assets/envs/success/robotcube_v0.gif){ width="120" loading=lazy } | `RobotCube-v0` | MuJoCo | UR10e and Robotiq 2F-85 cube pick-and-place *(original)* | [notebook](../examples/envs/manipulation/robotcube_v0.ipynb) |
| ![RobotCubeSO101-v0 verified success](../assets/envs/success/robotcubeso101_v0.gif){ width="120" loading=lazy } | `RobotCubeSO101-v0` | MuJoCo | SO-101 cube pick-and-place *(original)* | [notebook](../examples/envs/manipulation/robotcubeso101_v0.ipynb) |
| ![SpatialArm-v0 verified success](../assets/envs/success/spatialarm_v0.gif){ width="120" loading=lazy } | `SpatialArm-v0` | MuJoCo | Three-axis spatial arm positioning | [notebook](../examples/envs/manipulation/spatialarm_v0.ipynb) |
| ![FrankaCubeLift-v0 verified success](../assets/envs/success/frankacubelift_v0.gif){ width="120" loading=lazy } | `FrankaCubeLift-v0` | Isaac Lab | Franka grasping and lifting a free cube | [notebook](../examples/envs/manipulation/frankacubelift_v0.ipynb) |
| ![FrankaObstacleReach-v0 verified success](../assets/envs/success/frankaobstaclereach_v0.gif){ width="120" loading=lazy } | `FrankaObstacleReach-v0` | Isaac Lab | Franka reaching around a tabletop barrier | [notebook](../examples/envs/manipulation/frankaobstaclereach_v0.ipynb) |
| ![FrankaTablePush-v0 verified success](../assets/envs/success/frankatablepush_v0.gif){ width="120" loading=lazy } | `FrankaTablePush-v0` | Isaac Lab | Franka contact pushing on a tabletop | [notebook](../examples/envs/manipulation/frankatablepush_v0.ipynb) |

## Add your own

Register an `EnvSpec` with a `category` from `CATEGORIES`, or a new name of your own, and it appears in `list_envs(category=...)`. See [Add an environment or task](../guides/benchmark-environments.md#add-an-environment-or-task).
