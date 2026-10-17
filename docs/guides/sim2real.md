# SO101 chess: simulation to real hardware

[`torch_world.sim2real`](../api/sim2real.md) trains action-conditioned world models on simulated chess-piece manipulation and runs the same Cartesian controller on an SO101 follower. Start with the single [experiment notebook](../examples/sim2real/so101_chess.ipynb).

Version one moves **one piece on an otherwise empty board**. It includes procedural pawn, rook, knight, bishop, queen, and king geometry in both colors. The models receive wrist RGB and, where supported, measured arm/gripper state. Only simulation collection and scoring use ground-truth piece poses. Training uses no real demonstrations or fine-tuning. A real goal photograph is an evaluation input.

## Install and run

```bash
uv pip install -e '.[sim2real,notebooks,test]'
```

For the hardware notebook kernel, use Python 3.12 or newer:

```bash
uv pip install -e '.[so101,notebooks]'
```

The hardware extra pins **LeRobot 0.6.1**. It uses `lerobot.robots.so_follower.SO101Follower`, degree-mode arm positions, and the gripper's separate 0–100 range. Simulation and mock tests do not require LeRobot or a serial device. Robot meshes are fetched from the existing pinned MuJoCo Menagerie revision on first use; pretrained DINOv2 weights are downloaded only for non-smoke patch-model recipes.

```python
from torch_world.sim2real import SO101Config, run_experiment

result = run_experiment(
    config=SO101Config(size=32),
    model="lewm",  # or "lewam", "dinowm", "robojepa"
    smoke=True,
    episodes=2,
    updates=2,
    collect_steps=12,
    evaluation_steps=4,
    evaluation_seeds=(10007,),
    output_dir="runs/so101",
)
```

The smoke configuration checks the workflow. It is deliberately too small to establish manipulation performance. For experiments, set `smoke=False`, use the default 84-pixel inputs, collect full episodes, increase the data/training budget, and evaluate at least 12 seeds to cover every piece/color. LeWM and LeWAM use trainable CNNs; LeWAM additionally learns action chunks. DINO-WM uses frozen DINOv2 patches; the RoboJEPA recipe is explicitly a **RoboJEPA/DINOv2 variant**, not the paper's V-JEPA encoder. `make_adapter` also accepts custom encoders with declared feature dimensions and patch grids.

## Match your physical setup before training

The notebook uses `reference_setup()` for the connected-camera scene: a 250 × 250 mm wooden case, cream/brown grain, brass catches, warm light, white arm, and 16:9 wrist images. The owner confirmed the photo shows the resting pose and supplied the overall size and 1 mm panel thickness. A fit to 18 grid intersections and two front corners estimates a 22.65 mm border, 25.59 mm squares, a 54.66° vertical field of view, and board placement in the base frame. The fit has a 3.65-pixel coordinate RMS residual on those same landmarks; this is not independent calibration validation. The case depth (15 mm), lens distortion, mounting, joint pose and piece geometry remain unverified estimates. Fit evidence is stored in `examples/sim2real/assets/rest-view-fit.json`.

The farther-forward board uses a radial approach 1.2 radians below horizontal, configured with `SO101Config.approach_pitch`; the default remains vertical for existing setups. This tilt is included in checkpoint/goal compatibility checks. The nominal pawn setup has 14 reachable squares. Joint, workspace and collision checks remain active. `render_reference_pose(scene)` temporarily displays nearly closed jaws in the fitted resting pose, then restores the complete integration state. Training and evaluation use their actual open-jaw working approach pose with the same board and camera. Presentation renders support the original 1280 × 720 resolution; model inputs use the physical adapter’s square resize.

Run a full, simulation-only experiment on CUDA with:

```bash
python -m torch_world.sim2real.run --episodes 72 --updates 3000 --eval-episodes 24
```

The CLI defaults to DINO-WM with pretrained DINOv2, 140-pixel inputs, and cached frozen visual features. It saves data and a checkpoint before evaluation, writes progress, and records execution settings and a progression gate. It imports no physical driver and never connects to a robot. Multiple machines can run independent seeds via `--seed`; the trainer is not distributed.

To compare a fresh model on an existing run's exact training/validation split, camera geometry, randomized domain, evaluation seeds and planning budget:

```bash
python -m torch_world.sim2real.retrain runs/so101-rest-view/RUN_ID \
    --model lewm --updates 3000 --output runs/so101-lewm-rest-view
```

This copies the saved dataset, trains new weights, and writes separate training and evaluation artifacts. The source must contain `episodes.pt`, `model.pt`, `results.json`, and the original CLI's `execution.json`. The LeWM adapter is a compact CNN baseline with a learned 32-dimensional visual representation; it is not the paper-scale ViT architecture. Compare held-out manipulation outcomes rather than raw loss values across different model objectives. The notebook contains the comparison; prediction diagnostics are archived in `examples/sim2real/reports/`.

The fitted-view LeWM comparison used the same 72 training episodes, 3,000 updates, and 24 evaluation seeds per domain. It completed **0/24 nominal and 0/24 randomized placements**, matching DINO-WM's placement result. Controller rejections decreased from 41/48 to 18/48, which did not establish manipulation success. The model swap alone did not solve the task; the shared demonstration quality, short planning horizon, and final-goal objective remain limitations. Physical trials remain disabled.

### LeWM with measured state and rollout training

Two opt-in changes address the compact baseline's missing robot state and one-step training objective:

```bash
python -m torch_world.sim2real.retrain runs/so101-rest-view/RUN_ID \
    --model lewm --proprioception --rollout-steps 8 \
    --updates 3000 --output runs/so101-lewm-state-rollout
```

The adapter appends five measured joint angles (divided by pi) and gripper opening to its learned visual latent. The predictor learns both visual and state futures; it never receives future measured states during rollout. State and visual MSE are averaged separately and added with equal weights. SIGReg applies only to visual features. Eight-step rollout MSE is added to the teacher-forced loss. A three-frame context therefore requires eleven frames per training window; windows remain inside their original episode and validation remains disjoint. The CNN and planner budgets stay unchanged. These options are saved in the checkpoint recipe; the original LeWM defaults and older checkpoints remain supported.

The Python equivalent is `make_adapter("lewm", size=140, proprioception=True, rollout_steps=8)`, passed as `adapter=` to `run_experiment`. Training logs report prediction, state, rollout and regularization terms separately. Do not interpret the total loss as directly comparable to the original LeWM objective.

The recorded Spark run completed **0/24 nominal and 0/24 randomized placements**. Recorded lift events increased from 2/48 to 6/48, while controller rejections increased from 18/48 to 25/48. This is not an improvement in completed-task performance. On 72 sampled held-out windows, eight-step state predictions beat persistence, but one-step joint MAE remained 6.23 degrees versus 1.16 degrees for persistence. The archived reports record this limitation; physical execution remains disabled.

### Separate intermediate-goal diagnostic

`StagedWorldModelPolicy` accepts fixed `GoalStage` images and measured arm/gripper states. It retains observation/action history across goal changes and advances after measured joints and opening stay near a stage target. Reaching a stage does not establish that a piece was grasped or placed; the original environment success conditions remain authoritative.

```bash
python -m torch_world.sim2real.staged runs/so101-lewm-state-rollout/training/RUN_ID \
    --source-report runs/so101-rest-view/RUN_ID/results.json \
    --output runs/so101-lewm-staged --samples 64 --iterations 3 --objective path
```

This evaluation obtains approach, grasp, lift, transfer, lower, release, clearance, withdrawal and return images from a separate scripted execution of each task. It then resets a separate scene for learned control using only fixed goals and live observations. The supplied intermediate references are **scripted assistance**, not an autonomous final-image policy. Failed reference attempts remain in the full evaluation denominator and are labeled separately from controller rejections. Goal tensors and phase transitions are saved. This diagnostic cannot pass the physical progression gate, even if every assisted trial succeeds. It never connects to hardware.

The example also increases CEM to 64 samples and three iterations and averages goal cost along the predicted path. It tests goals and additional search together; it cannot isolate their separate effects. Omit the three overrides to retain the source run's original final-state planning budget. Execution reports record these settings.

The recorded staged experiment completed 0/24 placements in each domain. Successful references were available for 17 nominal and 12 randomized tasks; the other 19 tasks remain recorded as reference failures. Of 29 executed learned trials, 28 ended at approach and one reached the lift stage. The additional goals and search did not resolve the tracking problem. The archived summary and phase logs are in `examples/sim2real/reports/spark-lewm-staged.json`.

The recorded Spark experiment (72 training episodes, 3,000 updates, DINOv2 ViT-S/14 features) completed 0/12 nominal and 0/12 randomized learned-policy trials. Scripted baselines completed 4/12 and 2/12. The archived JSON includes controller failures, timing and progression-gate reasons. **That experiment used the previous camera transform.** Its checkpoint is incompatible with the corrected view; retrain after final camera calibration. A fresh fitted-view experiment is recorded separately in `reports/spark-dinowm-rest-view.json`, with 72 training episodes, 3,000 updates and 24 evaluation trials per domain. The fitted-view learned controller completed 0/24 nominal and 0/24 randomized placements; scripted baselines completed 17/24 and 12/24. The gate failed. The notebook displays the report along with the corrected local scripted and smoke runs. Physical execution remains disabled.

The default planning budget is an 8-step horizon, 16 candidates, and one CEM iteration. A short warm benchmark on the shared Spark measured 60–76 ms per call; this is not a guaranteed deadline or evidence of manipulation success. `--horizon`, `--samples`, and `--iterations` expose the compute/plan-quality tradeoff. Evaluate another budget without recollecting or retraining:

```bash
python -m torch_world.sim2real.run --checkpoint-run runs/so101-rest-view/RUN_ID \
  --output runs/so101-evaluation --eval-episodes 24 --horizon 8 --samples 16 --iterations 1
```

This writes a new evaluation directory, keeps the original training artifacts, and rejects evaluation seeds that overlap training or validation. `restore_run` and `evaluate_saved_run` in `torch_world.sim2real.evaluation` expose the same workflow to Python.

`simulation_gate(result)` checks learned-policy placements on held-out nominal/randomized trials, controller failures, and planning latency. Its default minimum is 24 trials per domain, with 90% nominal and 80% randomized success. The default full run evaluates 24 trials per domain to meet the sample-count requirement. The notebook keeps physical actuation disabled and also requires this gate to pass before its physical-trial cell can run. Passing still does not replace physical calibration or establish real-world success.

The arm's printed parts default to white; `SO101Config(arm_color=(r, g, b))` changes their RGB appearance while retaining dark servos. Model input resolution (`size`) is independent of presentation quality: `scripted_episode(render_size=512)` returns native-resolution wrist images in `views` and a reference video. Evaluation videos render at 384 pixels by default. These renders use multisample antialiasing, rather than enlarging the training images.

For a camera-only inspection with no motor connection, install FFmpeg on PATH and use an explicit device:

```python
from torch_world.sim2real import capture_wrist_camera

report = capture_wrist_camera(
    "runs/wrist.png", device="icspring camera", width=1280, height=720, warmup=2.0,
)
```

On macOS the device is an AVFoundation camera name or index; on Linux use a V4L2 path such as `/dev/video2`. This helper saves an unmodified PNG plus a JSON report containing acquisition time, actual resolution, and image statistics. It never falls back to another camera or connects audio or a motor driver. The notebook includes a dated real-camera image and observation report; refreshing it is disabled by default so ordinary notebook execution remains motor-free and does not require a connected camera.

The default 32 mm squares and piece dimensions are examples. Replace them with measurements of your set using `ChessTaskConfig` and `PieceSpec`. `board_origin` is the centre of a1 in robot-base coordinates, and `board_yaw` rotates files/ranks about the vertical axis. Only reachable squares are sampled; a full-sized board is not assumed to fit the SO101 workspace. Procedural piece silhouettes are approximations of a set, not scanned assets.

1. Complete [LeRobot SO101 motor calibration](https://huggingface.co/docs/lerobot/so101). Use the same robot ID and calibration directory in every run. This calibrates the motors, but does not establish simulator joint signs/zeros.
2. Measure at least three corresponding simulator/physical poses spanning each joint by at least 10 degrees. `fit_joint_mapping(sim_radians, driver_degrees)` fits signs and offsets and reports per-joint residuals. Validate held-out poses before setting `Calibration.verified=True`.
3. Calibrate the wrist camera at its actual native resolution with `calibrate_camera(object_points, image_points, resolution)`. Supply at least five checkerboard views and inspect the returned reprojection error. Keep RGB channel order, camera mounting, resolution, and lens settings fixed.
4. Obtain paired `base_from_wrist` and `camera_from_board` 4×4 transforms at five or more diverse arm poses, with the board stationary. `calibrate_hand_eye` estimates `wrist_from_camera`. Transforms map coordinates from the name after `from` into the frame before it. Camera axes follow OpenCV: x right, y down, z forward. The wrist frame is Menagerie's `gripperframe` site. Board corner correspondences must have a consistent orientation; an unmarked chessboard is rotationally ambiguous.
5. `board_pose` composes the board-to-base transform. Translate its origin to the centre of a1 before filling `ChessTaskConfig.board_origin`; use the planar yaw and table height. This task assumes a horizontal board. Check rendered and real images with `overlay`; verify both arm alignment and square locations across held-out poses.
6. Measure the gripper's open/closed endpoints and each piece's dimensions, mass, and graspable neck. Save the `Calibration` JSON and set `task.measured=True` only after entering real measurements.

Load the measured profile **before simulation training** and pass `calibration=` to both `SO101Chess` and `run_experiment`. The camera model is rendered with calibrated intrinsics/extrinsics, and real frames are undistorted and resized consistently. Checkpoints bind camera calibration, task geometry, image size, and action/control settings. Changing these settings requires a matching experiment, rather than silently reusing incompatible weights.

## Actions, collection, and scoring

Actions have four components in [-1,1]: Cartesian x/y/z increments in the robot base frame and incremental opening (positive opens). Defaults are 5 mm per translational action, 0.15 radians per joint command, 0.15 gripper opening per step, and a 100 ms simulation control interval. Joint IK, workspace/table-clearance checks, and limits are shared across both backends. Hardware logs the actual command returned by LeRobot, which may limit it further.

Simulation data has T actions and T+1 observations, with measured proprioception and acquisition timestamps. `EpisodeWindows` aligns the simulator’s constant observation delay with applied actions, drops unobserved tail transitions, never crosses a reset, and adds the unused final action only when calling existing model losses. Cached features use the same aligned frame indices. Collection balances piece types and colors, perturbs a scripted grasp controller, and retains failed episodes with their outcomes. Randomization covers geometry, mass, friction, actuator gains, lighting, appearance, camera mounting, pixel noise, and discrete action/observation delays. Episode parameters and seeds are saved.

Success requires an upright, released piece inside its target square, with clearance for the entire base, the arm withdrawn to its reference pose, and a stable placement for several steps. Merely reaching the target while holding the piece does not count. Nominal and randomized evaluations report learned, scripted, and random controllers on the same seeds, including controller rejections and time limits. The scripted controller is a data/reference controller and is never substituted for a learned physical policy.

## Physical trials

Constructing or importing `SO101Robot` never connects. Its default is an in-memory mock and `dry_run=True`; `hardware=True, dry_run=False` plus a port, verified calibration, and measured geometry explicitly selects the real driver. `reset()` reads the current arm pose and resets controller history; it never homes or resets the physical scene. LeRobot's own connection setup can configure motors and toggle torque; connect with the arm supported as required by its driver.

Capture a goal by manually placing the desired piece on the target square and positioning the open arm at the reference view. `save_goal` stores RGB, measured joints, opening, and the calibration contract. Reset the scene manually to the source square; the goal image must show the destination clearly from the chosen reference pose. Avoid goals with the piece outside the wrist camera's field of view.

```python
from torch_world.sim2real import SO101Robot, WorldModelPolicy, load_checkpoint, load_goal, run_trial

adapter = load_checkpoint("runs/my_run/model.pt", config, task, calibration)
goal = load_goal("runs/my_goal.pt", config, task, calibration)
policy = WorldModelPolicy(adapter, goal)
with SO101Robot(config, task, calibration, port=PORT,
                hardware=True, dry_run=False) as robot:
    trial = run_trial(robot, policy, steps=120, output_dir="runs/so101-real")
```

A motor-free latency check rejects planners that exceed the control interval. Actual trials also reject late planning or stale camera observations. Dispatch is rate-limited; acquisition, inference, and serial overhead are logged rather than presented as hard real-time control. Select budgets on the deployment computer and include appropriate delays in simulation training.

Interrupting a trial stops dispatch and attempts a measured-position hold. Closing preserves torque; `release_torque()` is an explicit action after supporting the arm/object. A lost serial connection may prevent confirming a hold; the error is surfaced. Joint/table checks sample a known static scene and are not a general collision planner for people or clutter.

Call `label_trial(trial, True/False, notes=...)` after observing the real outcome. Mock runs cannot be labeled as physical successes. Failed or interrupted trials retain their logs and captured frames. No physical success rate is manufactured from simulation performance.

## Artifacts and extension points

Each experiment creates a unique directory containing `model.pt`, `episodes.pt`, `results.json`, `videos.pt`, optional GIFs, and a completion manifest. Tensor artifacts load with `weights_only=True`. Keep calibration JSON and goal photos with the experiment. The exact checkpoint compatibility fingerprint includes the seed/configuration, so load using the saved settings.

Use `ModelAdapter` for model-specific losses/encodings, `WorldModelPolicy` for observation/action history with CEM or LeWAM proposals, and `SO101Controller` for shared kinematics. Custom encoders require a matching adapter when reloading a checkpoint. `MockDriver` exercises returned-command limiting, stale observations, connection errors, and cleanup without importing LeRobot.

Sources: [LeRobot 0.6.1](https://github.com/huggingface/lerobot/tree/v0.6.1), [MuJoCo camera conventions](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera), and the existing [MuJoCo Menagerie SO101](https://github.com/google-deepmind/mujoco_menagerie/tree/0059d4335f8156206f63a35662313385f7ad6d74/robotstudio_so101).


## LeWAM: jointly learned actions and dynamics

`make_adapter("lewam", proprioception=True, chunk_size=8)` constructs the new
`torch_world.models.LeWAM` model with the same SO101 image/action contract.
Use `LeWAMPlanner()` for the direct flow-matching policy, or
`LeWAMPlanner(num_samples=16)` to rank learned action proposals with the latent
world model. The shared `WorldModelPolicy` executes one action per observation,
records the applied command, and repeats initial observations to match training.
Pass `action_scale=(1, 1, 1, config.max_gripper_step)` when using a custom control
configuration. Existing LeWM checkpoints and CEM remain supported.

The LeWAM dataset adapter aligns the simulator's constant delayed observations
with applied commands, drops unobserved tail transitions, and supplies terminal
images from the same episode as goals. Only successful demonstrations supervise
its action branch; failures remain dynamics examples. Missing success labels do
not count as successful demonstrations. Training warns if no action supervision
is available, which is normal for a short smoke run.

The [dedicated LeWAM notebook](../examples/action/lewam.ipynb) covers the library's
arm benchmark. The [SO101 notebook](../examples/sim2real/so101_chess.ipynb) covers
chess simulation, the recorded wrist view, model comparison, and physical
calibration. Investigation, curriculum and expert-correction experiments live
in reusable `torch_world.sim2real` modules and run only when explicitly invoked.

See [LeWAM model scope](../models/jepa/lewam.md) for the implemented reference
variant, measured-state extension and differences from upstream.
`torch_world.sim2real.diagnostics.diagnose_lewam` reports held-out latent/action
errors against persistence and zero-action baselines without changing weights.


The recorded Spark LeWAM run trained for 3,000 updates on the prior 72/18 episode
split. It completed **0/24 nominal and 0/24 randomized placements**, with no lift
events and clearance rejection in every learned-policy trial. Eight-step latent
and joint-state prediction beat persistence, but one-step prediction and sampled
action errors did not. See `examples/sim2real/reports/spark-lewam.json` for exact
metrics and provenance. This checkpoint does not pass the simulation gate.

Long-lived Spark evaluation processes stalled inside native EGL rendering and
cleanup. The completed nominal evaluation was retained; all randomized learned
trials were evaluated again in a short process. Randomized scripted/random
controls are explicitly reused from the matching source experiment. No partial
learned trials were pooled. `evaluate(..., policies=("learned",))` selects a
policy for independent evaluation; preserve the full ordered seed list to keep
piece/color assignments identical. `evaluate_saved_run` now also writes every
completed outcome to `episodes.jsonl`, retaining evidence if a native renderer
later stalls.

### Investigating a failed checkpoint

Run a fixed pawn move with randomization and delays disabled before collecting
another large dataset:

```sh
python -m torch_world.sim2real.investigation runs/so101-lewam/training/RUN_ID \
  --source b2 --target a5 --updates 2000 --output runs/lewam-investigation
```

The output must be a new directory. `--updates 0` performs only the checkpoint,
replay and contact audit. The full command additionally trains a fresh compact
model on one successful demonstration, then compares rendered and actual
terminal-image goals with three policy seeds. Those repeated executions share
one scene and are not independent held-out tasks. Original checkpoint weights,
controller bounds and success criteria are preserved.

`trace_episode` retains every attempted command and any `ClearanceError`
evidence: geometry/body names, penetration distance and sampled path fraction.
It compares the existing requested-opening check with interpolated measured
opening without advancing physics. The controller still uses its original
clearance rule; diagnostic evidence is not an exception to collision checking.
`action_tracking` reports teacher-forced first-action errors and an active-motion
subset, alongside the zero-action baseline. A good supervised action fit must
still be tested in closed loop, where the policy changes its own observations.

## Review fixes and archived results

The archived reports predate three review fixes: non-LeWAM datasets now align
delayed observations and applied commands, calibrated renders use the camera’s
native aspect ratio, and the expert withdraws clear of wide piece heads before
lifting. These changes require fresh training and evaluation; existing JSON
results are retained as historical evidence, not results for the corrected code.
Neither the smoke checks nor the historical learned policies qualify for hardware.
