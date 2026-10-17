# Environment notebooks

LeWM is the default; replace `model_factory` to compare another WorldModel.

Execution status is shown for every notebook. Executed outputs are small validation runs, not tuned benchmark scores.

See [setup and protocol](https://github.com/latent-horizon/pytorch-world/blob/main/docs/guides/benchmark-environments.md).

[Custom model across three environments](custom_model_three_envs.ipynb): compare a small residual world model with LeWM using one shared protocol.

[Original environment results: Macaw, PointMass, PushT and RobotCube](LEGACY_RESULTS.md).

[Rendering quality comparison](render_quality_comparison.ipynb): three presets on Maze, BallPlate, and RGB FrankaCubeLift.

| Notebook | Backend | Status | Task |
|---|---|---|---|
| [Maze-v0](01_maze_v0.ipynb) | torch | Executed | Route around walls and dead ends |
| [KeyDoor-v0](02_keydoor_v0.ipynb) | torch | Executed | Collect a key before crossing a locked door |
| [Sokoban-v0](03_sokoban_v0.ipynb) | torch | Executed | Push a box to a target without trapping it |
| [FrankaCubeLift-v0](04_frankacubelift_v0.ipynb) | isaaclab | Executed | Franka grasping and lifting a free cube |
| [FrankaObstacleReach-v0](05_frankaobstaclereach_v0.ipynb) | isaaclab | Executed | Franka reaching around a tabletop barrier |
| [FrankaTablePush-v0](06_frankatablepush_v0.ipynb) | isaaclab | Executed | Franka contact pushing on a tabletop |
| [BallPlate-v0](07_ballplate_v0.ipynb) | mujoco | Executed | Tilt a plate to position a constrained ball |
| [BlockStack-v0](08_blockstack_v0.ipynb) | mujoco | Executed | Stack a free cube on a raised block with a Robotiq 2F-85 |
| [BoxGate-v0](09_boxgate_v0.ipynb) | mujoco | Executed | Push a box through a narrow gate |
| [CartpoleSwingup-v0](10_cartpoleswingup_v0.ipynb) | mujoco | Executed | Torque-limited cart-pole swing-up |
| [DifferentialParking-v0](11_differentialparking_v0.ipynb) | mujoco | Executed | Park a differential-drive robot between barriers |
| [DoorOpen-v0](12_dooropen_v0.ipynb) | mujoco | Executed | Push a hinged door open |
| [DoubleCartpole-v0](13_doublecartpole_v0.ipynb) | mujoco | Executed; reference pending | Underactuated double-pole stabilization |
| [DrawerOpen-v0](14_draweropen_v0.ipynb) | mujoco | Executed | Contact a handle and pull a sliding drawer |
| [HurdleHopper-v0](15_hurdlehopper_v0.ipynb) | mujoco | Executed | Hop a single articulated leg across a hurdle |
| [ObstacleArm-v0](16_obstaclearm_v0.ipynb) | mujoco | Executed | Two-link arm positioning around an obstacle |
| [PayloadTransport-v0](17_payloadtransport_v0.ipynb) | mujoco | Executed | Transport a suspended payload without swing |
| [PegInsertion-v0](18_peginsertion_v0.ipynb) | mujoco | Executed | Insert a Cartesian peg into a clearance fixture |
| [PickPlace-v0](19_pickplace_v0.ipynb) | mujoco | Executed | Grasp and relocate a free cube with a Robotiq 2F-85 |
| [PlanarQuadrotor-v0](20_planarquadrotor_v0.ipynb) | mujoco | Executed | Differential-thrust planar flight |
| [PuckPush-v0](21_puckpush_v0.ipynb) | mujoco | Executed | Contact-only planar object pushing |
| [PuckSlide-v0](22_puckslide_v0.ipynb) | mujoco | Executed | Slide an object beyond the tool workspace by releasing momentum |
| [ReactionWheel-v0](23_reactionwheel_v0.ipynb) | mujoco | Executed | Reaction-wheel pendulum balance |
| [SpatialArm-v0](24_spatialarm_v0.ipynb) | mujoco | Executed | Three-axis spatial arm positioning |
| [NewtonAcrobot-v0](25_newtonacrobot_v0.ipynb) | newton | Executed | Underactuated two-link swing-up with Newton physics |
| [NewtonPendulum-v0](26_newtonpendulum_v0.ipynb) | newton | Executed | Torque-limited swing-up with Newton physics |
| [Checkpoints-v0](27_checkpoints_v0.ipynb) | torch | Executed | Visit two checkpoints in order before reaching the exit |
| [IceMaze-v0](28_icemaze_v0.ipynb) | torch | Executed | Slide until blocked; plan stopping positions |
| [PushT-v0](29_pusht_v0.ipynb) | mujoco | Executed | Push a T-shaped block to a fixed target pose |
| [RobotCube-v0](30_robotcube_v0.ipynb) | mujoco | Executed | UR10e and Robotiq 2F-85 cube pick-and-place |
| [RobotCubeSO101-v0](31_robotcubeso101_v0.ipynb) | mujoco | Executed | SO-101 cube pick-and-place |
| [Macaw-v0](32_macaw_v0.ipynb) | torch | Executed | Fly the animated macaw to a target position |
| [PointMass-v0](33_pointmass_v0.ipynb) | torch | Executed | Move a point mass to a target in the unit square |

## Generate results

Run all environment notebooks in fresh kernels with the shared LeWM/CEM protocol:
```bash
python examples/envs/run_notebooks.py --output-dir runs/benchmarks/my-campaign
```
Pass `--backends torch mujoco` to select backends, or `--environments BlockStack-v0 PickPlace-v0` to select tasks. Use `--physics-python /path/to/isaaclab/.venv/bin/python` for Isaac Lab/Newton notebooks; that runtime also needs `ipykernel`. Output paths are relative to this directory.
Repeat the same command to retry failed notebooks and skip completed ones. Each campaign contains `execution.json`, `summary.csv`, and versioned per-environment runs with metrics, provenance, GT and episodes. Every episode GIF includes GT. Provenance hashes include packaged hardware models and meshes. Keep different library hashes and task configurations separate when comparing campaigns.

Notebooks retain static episode previews and measured results. Rerun a cell to view its animation. After executing notebooks, use `python examples/envs/archive_media.py` to save animations locally and keep only previews in notebook outputs. `media/` and large runtime datasets under `runs/` are gitignored.
