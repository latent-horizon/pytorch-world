---
title: PyTorch World
titleTemplate: false
pageClass: tw-home
sidebar: false
aside: false
outline: false
prev: false
next: false
---

<div class="tw-hero">

![pytorch-world](assets/banner-dark.png)

World models for PyTorch. Learn how a world evolves, then plan by imagining what comes next. Shared parts keep every model short and easy to compare.

<div class="tw-install">

```bash
uv add torch_world
```

</div>

<div class="tw-actions">

[Get started](getting-started/quickstart.md){ .tw-button .tw-button--primary }
[Models](models/index.md){ .tw-button }
[API](api/index.md){ .tw-button }

</div>

</div>


<div class="grid cards">

-   :material-family-tree:{ .lg } **Model families**

    ---

    Models grouped by how they learn: JEPA, from image pretraining to multi-robot control; DINO; TD-MPC, for reward-driven control; and world action models, which imagine video and act. More families are coming.

    [:octicons-arrow-right-24: Models](models/index.md)

-   :material-earth:{ .lg } **One world-model interface**

    ---

    `encode`, `predict` and `rollout` are the same for every world model, so they can be trained, compared and planned with interchangeably.

    [:octicons-arrow-right-24: Concepts](getting-started/concepts.md)

-   :material-map-marker-path:{ .lg } **Planning**

    ---

    Search for actions inside a model's imagination. The cross-entropy method works with any world model, with no extra code.

    [:octicons-arrow-right-24: Planning guide](guides/planning.md)

-   :material-robot-industrial:{ .lg } **Environments**

    ---

    190 environments: 33 built-in tasks from planning puzzles to Franka manipulation, plus OGBench, DMControl, Meta-World, ManiSkill, MyoSuite, RoboCasa and RoboDojo.

    [:octicons-arrow-right-24: Environments](environments/index.md)

-   :material-database-outline:{ .lg } **Datasets**

    ---

    Windows over collected experience for training dynamics, plus medical images and questions for vision-language models.

    [:octicons-arrow-right-24: Datasets](datasets/index.md)

-   :material-cube-outline:{ .lg } **Building blocks**

    ---

    Vision transformers for images and video, conditioned predictors, masking strategies and objectives, shared by every family.

    [:octicons-arrow-right-24: API reference](api/index.md)

</div>

## Environments

190 environments: 33 built-in, 13 from OGBench, 20 from DMControl, 10 from Meta-World, 12 from ManiSkill, 10 from MyoSuite, 50 from RoboCasa and 42 from RoboDojo. Built-in tasks show a verified successful episode; OGBench tasks replay an episode of their dataset, Meta-World tasks its scripted expert, DMControl, ManiSkill and MyoSuite tasks a random policy, RoboCasa tasks a human demonstration, and RoboDojo tasks its official task video. See [Environments](environments/index.md) for each one.

### Built-in

<div class="tw-carousel">

[![Checkpoints-v0](assets/envs/success/checkpoints_v0.gif){ loading=lazy width="120" height="120" }<span>Checkpoints</span>](examples/envs/puzzles/checkpoints_v0.ipynb)
[![IceMaze-v0](assets/envs/success/icemaze_v0.gif){ loading=lazy width="120" height="120" }<span>IceMaze</span>](examples/envs/puzzles/icemaze_v0.ipynb)
[![KeyDoor-v0](assets/envs/success/keydoor_v0.gif){ loading=lazy width="120" height="120" }<span>KeyDoor</span>](examples/envs/puzzles/keydoor_v0.ipynb)
[![Maze-v0](assets/envs/success/maze_v0.gif){ loading=lazy width="120" height="120" }<span>Maze</span>](examples/envs/puzzles/maze_v0.ipynb)
[![Sokoban-v0](assets/envs/success/sokoban_v0.gif){ loading=lazy width="120" height="120" }<span>Sokoban</span>](examples/envs/puzzles/sokoban_v0.ipynb)
[![BallPlate-v0](assets/envs/success/ballplate_v0.gif){ loading=lazy width="120" height="120" }<span>BallPlate</span>](examples/envs/control/ballplate_v0.ipynb)
[![CartpoleSwingup-v0](assets/envs/success/cartpoleswingup_v0.gif){ loading=lazy width="120" height="120" }<span>CartpoleSwingup</span>](examples/envs/control/cartpoleswingup_v0.ipynb)
[![ReactionWheel-v0](assets/envs/success/reactionwheel_v0.gif){ loading=lazy width="120" height="120" }<span>ReactionWheel</span>](examples/envs/control/reactionwheel_v0.ipynb)
[![NewtonAcrobot-v0](assets/envs/success/newtonacrobot_v0.gif){ loading=lazy width="120" height="120" }<span>NewtonAcrobot</span>](examples/envs/control/newtonacrobot_v0.ipynb)
[![NewtonPendulum-v0](assets/envs/success/newtonpendulum_v0.gif){ loading=lazy width="120" height="120" }<span>NewtonPendulum</span>](examples/envs/control/newtonpendulum_v0.ipynb)
[![Macaw-v0](assets/envs/success/macaw_v0.gif){ loading=lazy width="120" height="120" }<span>Macaw</span>](examples/envs/navigation/macaw_v0.ipynb)
[![PointMass-v0](assets/envs/success/pointmass_v0.gif){ loading=lazy width="120" height="120" }<span>PointMass</span>](examples/envs/navigation/pointmass_v0.ipynb)
[![DifferentialParking-v0](assets/envs/success/differentialparking_v0.gif){ loading=lazy width="120" height="120" }<span>DifferentialParking</span>](examples/envs/navigation/differentialparking_v0.ipynb)
[![HurdleHopper-v0](assets/envs/success/hurdlehopper_v0.gif){ loading=lazy width="120" height="120" }<span>HurdleHopper</span>](examples/envs/navigation/hurdlehopper_v0.ipynb)
[![PayloadTransport-v0](assets/envs/success/payloadtransport_v0.gif){ loading=lazy width="120" height="120" }<span>PayloadTransport</span>](examples/envs/navigation/payloadtransport_v0.ipynb)
[![PlanarQuadrotor-v0](assets/envs/success/planarquadrotor_v0.gif){ loading=lazy width="120" height="120" }<span>PlanarQuadrotor</span>](examples/envs/navigation/planarquadrotor_v0.ipynb)
[![BlockStack-v0](assets/envs/success/blockstack_v0.gif){ loading=lazy width="120" height="120" }<span>BlockStack</span>](examples/envs/manipulation/blockstack_v0.ipynb)
[![BoxGate-v0](assets/envs/success/boxgate_v0.gif){ loading=lazy width="120" height="120" }<span>BoxGate</span>](examples/envs/manipulation/boxgate_v0.ipynb)
[![DoorOpen-v0](assets/envs/success/dooropen_v0.gif){ loading=lazy width="120" height="120" }<span>DoorOpen</span>](examples/envs/manipulation/dooropen_v0.ipynb)
[![DrawerOpen-v0](assets/envs/success/draweropen_v0.gif){ loading=lazy width="120" height="120" }<span>DrawerOpen</span>](examples/envs/manipulation/draweropen_v0.ipynb)
[![ObstacleArm-v0](assets/envs/success/obstaclearm_v0.gif){ loading=lazy width="120" height="120" }<span>ObstacleArm</span>](examples/envs/manipulation/obstaclearm_v0.ipynb)
[![PegInsertion-v0](assets/envs/success/peginsertion_v0.gif){ loading=lazy width="120" height="120" }<span>PegInsertion</span>](examples/envs/manipulation/peginsertion_v0.ipynb)
[![PickPlace-v0](assets/envs/success/pickplace_v0.gif){ loading=lazy width="120" height="120" }<span>PickPlace</span>](examples/envs/manipulation/pickplace_v0.ipynb)
[![PuckPush-v0](assets/envs/success/puckpush_v0.gif){ loading=lazy width="120" height="120" }<span>PuckPush</span>](examples/envs/manipulation/puckpush_v0.ipynb)
[![PuckSlide-v0](assets/envs/success/puckslide_v0.gif){ loading=lazy width="120" height="120" }<span>PuckSlide</span>](examples/envs/manipulation/puckslide_v0.ipynb)
[![PushT-v0](assets/envs/success/pusht_v0.gif){ loading=lazy width="120" height="120" }<span>PushT</span>](examples/envs/manipulation/pusht_v0.ipynb)
[![RobotCube-v0](assets/envs/success/robotcube_v0.gif){ loading=lazy width="120" height="120" }<span>RobotCube</span>](examples/envs/manipulation/robotcube_v0.ipynb)
[![RobotCubeSO101-v0](assets/envs/success/robotcubeso101_v0.gif){ loading=lazy width="120" height="120" }<span>RobotCubeSO101</span>](examples/envs/manipulation/robotcubeso101_v0.ipynb)
[![SpatialArm-v0](assets/envs/success/spatialarm_v0.gif){ loading=lazy width="120" height="120" }<span>SpatialArm</span>](examples/envs/manipulation/spatialarm_v0.ipynb)
[![FrankaCubeLift-v0](assets/envs/success/frankacubelift_v0.gif){ loading=lazy width="120" height="120" }<span>FrankaCubeLift</span>](examples/envs/manipulation/frankacubelift_v0.ipynb)
[![FrankaObstacleReach-v0](assets/envs/success/frankaobstaclereach_v0.gif){ loading=lazy width="120" height="120" }<span>FrankaObstacleReach</span>](examples/envs/manipulation/frankaobstaclereach_v0.ipynb)
[![FrankaTablePush-v0](assets/envs/success/frankatablepush_v0.gif){ loading=lazy width="120" height="120" }<span>FrankaTablePush</span>](examples/envs/manipulation/frankatablepush_v0.ipynb)

</div>

### OGBench

<div class="tw-carousel">

[![pointmaze-medium-navigate-v0](assets/envs/ogbench/pointmaze-medium-navigate.gif){ loading=lazy width="120" height="120" }<span>pointmaze-medium-navigate</span>](environments/index.md#ogbench)
[![antmaze-medium-navigate-v0](assets/envs/ogbench/antmaze-medium-navigate.gif){ loading=lazy width="120" height="120" }<span>antmaze-medium-navigate</span>](environments/index.md#ogbench)
[![visual-antmaze-medium-navigate-v0](assets/envs/ogbench/visual-antmaze-medium-navigate.gif){ loading=lazy width="120" height="120" }<span>visual-antmaze-medium-navigate</span>](environments/index.md#ogbench)
[![humanoidmaze-medium-navigate-v0](assets/envs/ogbench/humanoidmaze-medium-navigate.gif){ loading=lazy width="120" height="120" }<span>humanoidmaze-medium-navigate</span>](environments/index.md#ogbench)
[![antsoccer-arena-navigate-v0](assets/envs/ogbench/antsoccer-arena-navigate.gif){ loading=lazy width="120" height="120" }<span>antsoccer-arena-navigate</span>](environments/index.md#ogbench)
[![cube-single-play-v0](assets/envs/ogbench/cube-single-play.gif){ loading=lazy width="120" height="120" }<span>cube-single-play</span>](environments/index.md#ogbench)
[![visual-cube-single-play-v0](assets/envs/ogbench/visual-cube-single-play.gif){ loading=lazy width="120" height="120" }<span>visual-cube-single-play</span>](environments/index.md#ogbench)
[![cube-double-play-v0](assets/envs/ogbench/cube-double-play.gif){ loading=lazy width="120" height="120" }<span>cube-double-play</span>](environments/index.md#ogbench)
[![visual-cube-double-play-v0](assets/envs/ogbench/visual-cube-double-play.gif){ loading=lazy width="120" height="120" }<span>visual-cube-double-play</span>](environments/index.md#ogbench)
[![scene-play-v0](assets/envs/ogbench/scene-play.gif){ loading=lazy width="120" height="120" }<span>scene-play</span>](environments/index.md#ogbench)
[![visual-scene-play-v0](assets/envs/ogbench/visual-scene-play.gif){ loading=lazy width="120" height="120" }<span>visual-scene-play</span>](environments/index.md#ogbench)
[![puzzle-3x3-play-v0](assets/envs/ogbench/puzzle-3x3-play.gif){ loading=lazy width="120" height="120" }<span>puzzle-3x3-play</span>](environments/index.md#ogbench)
[![visual-puzzle-3x3-play-v0](assets/envs/ogbench/visual-puzzle-3x3-play.gif){ loading=lazy width="120" height="120" }<span>visual-puzzle-3x3-play</span>](environments/index.md#ogbench)

</div>

### DeepMind Control Suite

<div class="tw-carousel">

[![acrobot-swingup](assets/envs/dmcontrol/acrobot-swingup.gif){ loading=lazy width="120" height="120" }<span>acrobot-swingup</span>](environments/index.md#deepmind-control-suite)
[![ball_in_cup-catch](assets/envs/dmcontrol/ball_in_cup-catch.gif){ loading=lazy width="120" height="120" }<span>ball_in_cup-catch</span>](environments/index.md#deepmind-control-suite)
[![cartpole-balance](assets/envs/dmcontrol/cartpole-balance.gif){ loading=lazy width="120" height="120" }<span>cartpole-balance</span>](environments/index.md#deepmind-control-suite)
[![cartpole-balance_sparse](assets/envs/dmcontrol/cartpole-balance_sparse.gif){ loading=lazy width="120" height="120" }<span>cartpole-balance_sparse</span>](environments/index.md#deepmind-control-suite)
[![cartpole-swingup](assets/envs/dmcontrol/cartpole-swingup.gif){ loading=lazy width="120" height="120" }<span>cartpole-swingup</span>](environments/index.md#deepmind-control-suite)
[![cartpole-swingup_sparse](assets/envs/dmcontrol/cartpole-swingup_sparse.gif){ loading=lazy width="120" height="120" }<span>cartpole-swingup_sparse</span>](environments/index.md#deepmind-control-suite)
[![cheetah-run](assets/envs/dmcontrol/cheetah-run.gif){ loading=lazy width="120" height="120" }<span>cheetah-run</span>](environments/index.md#deepmind-control-suite)
[![finger-spin](assets/envs/dmcontrol/finger-spin.gif){ loading=lazy width="120" height="120" }<span>finger-spin</span>](environments/index.md#deepmind-control-suite)
[![finger-turn_easy](assets/envs/dmcontrol/finger-turn_easy.gif){ loading=lazy width="120" height="120" }<span>finger-turn_easy</span>](environments/index.md#deepmind-control-suite)
[![finger-turn_hard](assets/envs/dmcontrol/finger-turn_hard.gif){ loading=lazy width="120" height="120" }<span>finger-turn_hard</span>](environments/index.md#deepmind-control-suite)
[![hopper-hop](assets/envs/dmcontrol/hopper-hop.gif){ loading=lazy width="120" height="120" }<span>hopper-hop</span>](environments/index.md#deepmind-control-suite)
[![hopper-stand](assets/envs/dmcontrol/hopper-stand.gif){ loading=lazy width="120" height="120" }<span>hopper-stand</span>](environments/index.md#deepmind-control-suite)
[![pendulum-swingup](assets/envs/dmcontrol/pendulum-swingup.gif){ loading=lazy width="120" height="120" }<span>pendulum-swingup</span>](environments/index.md#deepmind-control-suite)
[![quadruped-run](assets/envs/dmcontrol/quadruped-run.gif){ loading=lazy width="120" height="120" }<span>quadruped-run</span>](environments/index.md#deepmind-control-suite)
[![quadruped-walk](assets/envs/dmcontrol/quadruped-walk.gif){ loading=lazy width="120" height="120" }<span>quadruped-walk</span>](environments/index.md#deepmind-control-suite)
[![reacher-easy](assets/envs/dmcontrol/reacher-easy.gif){ loading=lazy width="120" height="120" }<span>reacher-easy</span>](environments/index.md#deepmind-control-suite)
[![reacher-hard](assets/envs/dmcontrol/reacher-hard.gif){ loading=lazy width="120" height="120" }<span>reacher-hard</span>](environments/index.md#deepmind-control-suite)
[![walker-run](assets/envs/dmcontrol/walker-run.gif){ loading=lazy width="120" height="120" }<span>walker-run</span>](environments/index.md#deepmind-control-suite)
[![walker-stand](assets/envs/dmcontrol/walker-stand.gif){ loading=lazy width="120" height="120" }<span>walker-stand</span>](environments/index.md#deepmind-control-suite)
[![walker-walk](assets/envs/dmcontrol/walker-walk.gif){ loading=lazy width="120" height="120" }<span>walker-walk</span>](environments/index.md#deepmind-control-suite)

</div>

### Meta-World

<div class="tw-carousel">

[![reach-v3](assets/envs/metaworld/reach-v3.gif){ loading=lazy width="120" height="120" }<span>reach-v3</span>](environments/index.md#meta-world)
[![push-v3](assets/envs/metaworld/push-v3.gif){ loading=lazy width="120" height="120" }<span>push-v3</span>](environments/index.md#meta-world)
[![pick-place-v3](assets/envs/metaworld/pick-place-v3.gif){ loading=lazy width="120" height="120" }<span>pick-place-v3</span>](environments/index.md#meta-world)
[![door-open-v3](assets/envs/metaworld/door-open-v3.gif){ loading=lazy width="120" height="120" }<span>door-open-v3</span>](environments/index.md#meta-world)
[![drawer-open-v3](assets/envs/metaworld/drawer-open-v3.gif){ loading=lazy width="120" height="120" }<span>drawer-open-v3</span>](environments/index.md#meta-world)
[![drawer-close-v3](assets/envs/metaworld/drawer-close-v3.gif){ loading=lazy width="120" height="120" }<span>drawer-close-v3</span>](environments/index.md#meta-world)
[![button-press-topdown-v3](assets/envs/metaworld/button-press-topdown-v3.gif){ loading=lazy width="120" height="120" }<span>button-press-topdown-v3</span>](environments/index.md#meta-world)
[![peg-insert-side-v3](assets/envs/metaworld/peg-insert-side-v3.gif){ loading=lazy width="120" height="120" }<span>peg-insert-side-v3</span>](environments/index.md#meta-world)
[![window-open-v3](assets/envs/metaworld/window-open-v3.gif){ loading=lazy width="120" height="120" }<span>window-open-v3</span>](environments/index.md#meta-world)
[![window-close-v3](assets/envs/metaworld/window-close-v3.gif){ loading=lazy width="120" height="120" }<span>window-close-v3</span>](environments/index.md#meta-world)

</div>

### ManiSkill

<div class="tw-carousel">

[![PickCube-v1](assets/envs/maniskill/PickCube-v1.gif){ loading=lazy width="120" height="120" }<span>PickCube-v1</span>](environments/index.md#maniskill)
[![PushCube-v1](assets/envs/maniskill/PushCube-v1.gif){ loading=lazy width="120" height="120" }<span>PushCube-v1</span>](environments/index.md#maniskill)
[![PullCube-v1](assets/envs/maniskill/PullCube-v1.gif){ loading=lazy width="120" height="120" }<span>PullCube-v1</span>](environments/index.md#maniskill)
[![PokeCube-v1](assets/envs/maniskill/PokeCube-v1.gif){ loading=lazy width="120" height="120" }<span>PokeCube-v1</span>](environments/index.md#maniskill)
[![StackCube-v1](assets/envs/maniskill/StackCube-v1.gif){ loading=lazy width="120" height="120" }<span>StackCube-v1</span>](environments/index.md#maniskill)
[![PlaceSphere-v1](assets/envs/maniskill/PlaceSphere-v1.gif){ loading=lazy width="120" height="120" }<span>PlaceSphere-v1</span>](environments/index.md#maniskill)
[![RollBall-v1](assets/envs/maniskill/RollBall-v1.gif){ loading=lazy width="120" height="120" }<span>RollBall-v1</span>](environments/index.md#maniskill)
[![LiftPegUpright-v1](assets/envs/maniskill/LiftPegUpright-v1.gif){ loading=lazy width="120" height="120" }<span>LiftPegUpright-v1</span>](environments/index.md#maniskill)
[![PullCubeTool-v1](assets/envs/maniskill/PullCubeTool-v1.gif){ loading=lazy width="120" height="120" }<span>PullCubeTool-v1</span>](environments/index.md#maniskill)
[![PushT-v1](assets/envs/maniskill/PushT-v1.gif){ loading=lazy width="120" height="120" }<span>PushT-v1</span>](environments/index.md#maniskill)
[![PegInsertionSide-v1](assets/envs/maniskill/PegInsertionSide-v1.gif){ loading=lazy width="120" height="120" }<span>PegInsertionSide-v1</span>](environments/index.md#maniskill)
[![PlugCharger-v1](assets/envs/maniskill/PlugCharger-v1.gif){ loading=lazy width="120" height="120" }<span>PlugCharger-v1</span>](environments/index.md#maniskill)

</div>

### MyoSuite

<div class="tw-carousel">

[![myoFingerReachRandom-v0](assets/envs/myosuite/myoFingerReachRandom-v0.gif){ loading=lazy width="120" height="120" }<span>myoFingerReachRandom-v0</span>](environments/index.md#myosuite)
[![myoFingerPoseRandom-v0](assets/envs/myosuite/myoFingerPoseRandom-v0.gif){ loading=lazy width="120" height="120" }<span>myoFingerPoseRandom-v0</span>](environments/index.md#myosuite)
[![myoElbowPose1D6MRandom-v0](assets/envs/myosuite/myoElbowPose1D6MRandom-v0.gif){ loading=lazy width="120" height="120" }<span>myoElbowPose1D6MRandom-v0</span>](environments/index.md#myosuite)
[![myoHandReachRandom-v0](assets/envs/myosuite/myoHandReachRandom-v0.gif){ loading=lazy width="120" height="120" }<span>myoHandReachRandom-v0</span>](environments/index.md#myosuite)
[![myoHandPoseRandom-v0](assets/envs/myosuite/myoHandPoseRandom-v0.gif){ loading=lazy width="120" height="120" }<span>myoHandPoseRandom-v0</span>](environments/index.md#myosuite)
[![myoHandKeyTurnRandom-v0](assets/envs/myosuite/myoHandKeyTurnRandom-v0.gif){ loading=lazy width="120" height="120" }<span>myoHandKeyTurnRandom-v0</span>](environments/index.md#myosuite)
[![myoHandObjHoldRandom-v0](assets/envs/myosuite/myoHandObjHoldRandom-v0.gif){ loading=lazy width="120" height="120" }<span>myoHandObjHoldRandom-v0</span>](environments/index.md#myosuite)
[![myoHandPenTwirlRandom-v0](assets/envs/myosuite/myoHandPenTwirlRandom-v0.gif){ loading=lazy width="120" height="120" }<span>myoHandPenTwirlRandom-v0</span>](environments/index.md#myosuite)
[![myoHandReorient8-v0](assets/envs/myosuite/myoHandReorient8-v0.gif){ loading=lazy width="120" height="120" }<span>myoHandReorient8-v0</span>](environments/index.md#myosuite)
[![myoLegWalk-v0](assets/envs/myosuite/myoLegWalk-v0.gif){ loading=lazy width="120" height="120" }<span>myoLegWalk-v0</span>](environments/index.md#myosuite)

</div>

### RoboCasa

<div class="tw-carousel">

[![ArrangeBreadBasket](assets/envs/robocasa/ArrangeBreadBasket.gif){ loading=lazy width="120" height="120" }<span>Arrange Bread Basket</span>](environments/index.md#robocasa)
[![ArrangeTea](assets/envs/robocasa/ArrangeTea.gif){ loading=lazy width="120" height="120" }<span>Arrange Tea</span>](environments/index.md#robocasa)
[![BreadSelection](assets/envs/robocasa/BreadSelection.gif){ loading=lazy width="120" height="120" }<span>Bread Selection</span>](environments/index.md#robocasa)
[![CategorizeCondiments](assets/envs/robocasa/CategorizeCondiments.gif){ loading=lazy width="120" height="120" }<span>Categorize Condiments</span>](environments/index.md#robocasa)
[![CloseBlenderLid](assets/envs/robocasa/CloseBlenderLid.gif){ loading=lazy width="120" height="120" }<span>Close Blender Lid</span>](environments/index.md#robocasa)
[![CloseFridge](assets/envs/robocasa/CloseFridge.gif){ loading=lazy width="120" height="120" }<span>Close Fridge</span>](environments/index.md#robocasa)
[![CloseToasterOvenDoor](assets/envs/robocasa/CloseToasterOvenDoor.gif){ loading=lazy width="120" height="120" }<span>Close Toaster Oven Door</span>](environments/index.md#robocasa)
[![CoffeeSetupMug](assets/envs/robocasa/CoffeeSetupMug.gif){ loading=lazy width="120" height="120" }<span>Coffee Setup Mug</span>](environments/index.md#robocasa)
[![CuttingToolSelection](assets/envs/robocasa/CuttingToolSelection.gif){ loading=lazy width="120" height="120" }<span>Cutting Tool Selection</span>](environments/index.md#robocasa)
[![DeliverStraw](assets/envs/robocasa/DeliverStraw.gif){ loading=lazy width="120" height="120" }<span>Deliver Straw</span>](environments/index.md#robocasa)
[![GarnishPancake](assets/envs/robocasa/GarnishPancake.gif){ loading=lazy width="120" height="120" }<span>Garnish Pancake</span>](environments/index.md#robocasa)
[![GatherTableware](assets/envs/robocasa/GatherTableware.gif){ loading=lazy width="120" height="120" }<span>Gather Tableware</span>](environments/index.md#robocasa)
[![GetToastedBread](assets/envs/robocasa/GetToastedBread.gif){ loading=lazy width="120" height="120" }<span>Get Toasted Bread</span>](environments/index.md#robocasa)
[![HeatKebabSandwich](assets/envs/robocasa/HeatKebabSandwich.gif){ loading=lazy width="120" height="120" }<span>Heat Kebab Sandwich</span>](environments/index.md#robocasa)
[![KettleBoiling](assets/envs/robocasa/KettleBoiling.gif){ loading=lazy width="120" height="120" }<span>Kettle Boiling</span>](environments/index.md#robocasa)
[![LoadDishwasher](assets/envs/robocasa/LoadDishwasher.gif){ loading=lazy width="120" height="120" }<span>Load Dishwasher</span>](environments/index.md#robocasa)
[![MakeIceLemonade](assets/envs/robocasa/MakeIceLemonade.gif){ loading=lazy width="120" height="120" }<span>Make Ice Lemonade</span>](environments/index.md#robocasa)
[![NavigateKitchen](assets/envs/robocasa/NavigateKitchen.gif){ loading=lazy width="120" height="120" }<span>Navigate Kitchen</span>](environments/index.md#robocasa)
[![OpenCabinet](assets/envs/robocasa/OpenCabinet.gif){ loading=lazy width="120" height="120" }<span>Open Cabinet</span>](environments/index.md#robocasa)
[![OpenDrawer](assets/envs/robocasa/OpenDrawer.gif){ loading=lazy width="120" height="120" }<span>Open Drawer</span>](environments/index.md#robocasa)
[![OpenStandMixerHead](assets/envs/robocasa/OpenStandMixerHead.gif){ loading=lazy width="120" height="120" }<span>Open Stand Mixer Head</span>](environments/index.md#robocasa)
[![PackIdenticalLunches](assets/envs/robocasa/PackIdenticalLunches.gif){ loading=lazy width="120" height="120" }<span>Pack Identical Lunches</span>](environments/index.md#robocasa)
[![PanTransfer](assets/envs/robocasa/PanTransfer.gif){ loading=lazy width="120" height="120" }<span>Pan Transfer</span>](environments/index.md#robocasa)
[![PickPlaceCounterToCabinet](assets/envs/robocasa/PickPlaceCounterToCabinet.gif){ loading=lazy width="120" height="120" }<span>Pick Place Counter To Cabinet</span>](environments/index.md#robocasa)
[![PickPlaceCounterToStove](assets/envs/robocasa/PickPlaceCounterToStove.gif){ loading=lazy width="120" height="120" }<span>Pick Place Counter To Stove</span>](environments/index.md#robocasa)
[![PickPlaceDrawerToCounter](assets/envs/robocasa/PickPlaceDrawerToCounter.gif){ loading=lazy width="120" height="120" }<span>Pick Place Drawer To Counter</span>](environments/index.md#robocasa)
[![PickPlaceSinkToCounter](assets/envs/robocasa/PickPlaceSinkToCounter.gif){ loading=lazy width="120" height="120" }<span>Pick Place Sink To Counter</span>](environments/index.md#robocasa)
[![PickPlaceToasterToCounter](assets/envs/robocasa/PickPlaceToasterToCounter.gif){ loading=lazy width="120" height="120" }<span>Pick Place Toaster To Counter</span>](environments/index.md#robocasa)
[![PortionHotDogs](assets/envs/robocasa/PortionHotDogs.gif){ loading=lazy width="120" height="120" }<span>Portion Hot Dogs</span>](environments/index.md#robocasa)
[![PreSoakPan](assets/envs/robocasa/PreSoakPan.gif){ loading=lazy width="120" height="120" }<span>Pre Soak Pan</span>](environments/index.md#robocasa)
[![PrepareCoffee](assets/envs/robocasa/PrepareCoffee.gif){ loading=lazy width="120" height="120" }<span>Prepare Coffee</span>](environments/index.md#robocasa)
[![RecycleBottlesByType](assets/envs/robocasa/RecycleBottlesByType.gif){ loading=lazy width="120" height="120" }<span>Recycle Bottles By Type</span>](environments/index.md#robocasa)
[![RinseSinkBasin](assets/envs/robocasa/RinseSinkBasin.gif){ loading=lazy width="120" height="120" }<span>Rinse Sink Basin</span>](environments/index.md#robocasa)
[![ScrubCuttingBoard](assets/envs/robocasa/ScrubCuttingBoard.gif){ loading=lazy width="120" height="120" }<span>Scrub Cutting Board</span>](environments/index.md#robocasa)
[![SearingMeat](assets/envs/robocasa/SearingMeat.gif){ loading=lazy width="120" height="120" }<span>Searing Meat</span>](environments/index.md#robocasa)
[![SeparateFreezerRack](assets/envs/robocasa/SeparateFreezerRack.gif){ loading=lazy width="120" height="120" }<span>Separate Freezer Rack</span>](environments/index.md#robocasa)
[![SetUpCuttingStation](assets/envs/robocasa/SetUpCuttingStation.gif){ loading=lazy width="120" height="120" }<span>Set Up Cutting Station</span>](environments/index.md#robocasa)
[![SlideDishwasherRack](assets/envs/robocasa/SlideDishwasherRack.gif){ loading=lazy width="120" height="120" }<span>Slide Dishwasher Rack</span>](environments/index.md#robocasa)
[![StackBowlsCabinet](assets/envs/robocasa/StackBowlsCabinet.gif){ loading=lazy width="120" height="120" }<span>Stack Bowls Cabinet</span>](environments/index.md#robocasa)
[![SteamInMicrowave](assets/envs/robocasa/SteamInMicrowave.gif){ loading=lazy width="120" height="120" }<span>Steam In Microwave</span>](environments/index.md#robocasa)
[![StirVegetables](assets/envs/robocasa/StirVegetables.gif){ loading=lazy width="120" height="120" }<span>Stir Vegetables</span>](environments/index.md#robocasa)
[![StoreLeftoversInBowl](assets/envs/robocasa/StoreLeftoversInBowl.gif){ loading=lazy width="120" height="120" }<span>Store Leftovers In Bowl</span>](environments/index.md#robocasa)
[![TurnOffStove](assets/envs/robocasa/TurnOffStove.gif){ loading=lazy width="120" height="120" }<span>Turn Off Stove</span>](environments/index.md#robocasa)
[![TurnOnElectricKettle](assets/envs/robocasa/TurnOnElectricKettle.gif){ loading=lazy width="120" height="120" }<span>Turn On Electric Kettle</span>](environments/index.md#robocasa)
[![TurnOnMicrowave](assets/envs/robocasa/TurnOnMicrowave.gif){ loading=lazy width="120" height="120" }<span>Turn On Microwave</span>](environments/index.md#robocasa)
[![TurnOnSinkFaucet](assets/envs/robocasa/TurnOnSinkFaucet.gif){ loading=lazy width="120" height="120" }<span>Turn On Sink Faucet</span>](environments/index.md#robocasa)
[![WaffleReheat](assets/envs/robocasa/WaffleReheat.gif){ loading=lazy width="120" height="120" }<span>Waffle Reheat</span>](environments/index.md#robocasa)
[![WashFruitColander](assets/envs/robocasa/WashFruitColander.gif){ loading=lazy width="120" height="120" }<span>Wash Fruit Colander</span>](environments/index.md#robocasa)
[![WashLettuce](assets/envs/robocasa/WashLettuce.gif){ loading=lazy width="120" height="120" }<span>Wash Lettuce</span>](environments/index.md#robocasa)
[![WeighIngredients](assets/envs/robocasa/WeighIngredients.gif){ loading=lazy width="120" height="120" }<span>Weigh Ingredients</span>](environments/index.md#robocasa)

</div>

### RoboDojo

<div class="tw-carousel">

[![stack_bowls](assets/envs/robodojo/stack_bowls.gif){ loading=lazy width="120" height="120" }<span>stack bowls</span>](environments/index.md#robodojo)
[![push_T](assets/envs/robodojo/push_T.gif){ loading=lazy width="120" height="120" }<span>push T</span>](environments/index.md#robodojo)
[![pack_objects_into_box](assets/envs/robodojo/pack_objects_into_box.gif){ loading=lazy width="120" height="120" }<span>pack objects into box</span>](environments/index.md#robodojo)
[![fold_clothes](assets/envs/robodojo/fold_clothes.gif){ loading=lazy width="120" height="120" }<span>fold clothes</span>](environments/index.md#robodojo)
[![hang_mugs](assets/envs/robodojo/hang_mugs.gif){ loading=lazy width="120" height="120" }<span>hang mugs</span>](environments/index.md#robodojo)
[![sweep_blocks](assets/envs/robodojo/sweep_blocks.gif){ loading=lazy width="120" height="120" }<span>sweep blocks</span>](environments/index.md#robodojo)
[![pour_liquid_into_cup](assets/envs/robodojo/pour_liquid_into_cup.gif){ loading=lazy width="120" height="120" }<span>pour liquid into cup</span>](environments/index.md#robodojo)
[![make_toast](assets/envs/robodojo/make_toast.gif){ loading=lazy width="120" height="120" }<span>make toast</span>](environments/index.md#robodojo)
[![arrange_largest_number](assets/envs/robodojo/arrange_largest_number.gif){ loading=lazy width="120" height="120" }<span>arrange largest number</span>](environments/index.md#robodojo)
[![sort_nesting_dolls_by_size](assets/envs/robodojo/sort_nesting_dolls_by_size.gif){ loading=lazy width="120" height="120" }<span>sort nesting dolls by size</span>](environments/index.md#robodojo)
[![store_laptop_and_headphones](assets/envs/robodojo/store_laptop_and_headphones.gif){ loading=lazy width="120" height="120" }<span>store laptop and headphones</span>](environments/index.md#robodojo)
[![stack_blocks](assets/envs/robodojo/stack_blocks.gif){ loading=lazy width="120" height="120" }<span>stack blocks</span>](environments/index.md#robodojo)
[![fasten_screws](assets/envs/robodojo/fasten_screws.gif){ loading=lazy width="120" height="120" }<span>fasten screws</span>](environments/index.md#robodojo)
[![insert_tubes](assets/envs/robodojo/insert_tubes.gif){ loading=lazy width="120" height="120" }<span>insert tubes</span>](environments/index.md#robodojo)
[![plug_in_charger](assets/envs/robodojo/plug_in_charger.gif){ loading=lazy width="120" height="120" }<span>plug in charger</span>](environments/index.md#robodojo)
[![pour_balls_into_vase](assets/envs/robodojo/pour_balls_into_vase.gif){ loading=lazy width="120" height="120" }<span>pour balls into vase</span>](environments/index.md#robodojo)
[![play_Xylophone](assets/envs/robodojo/play_Xylophone.gif){ loading=lazy width="120" height="120" }<span>play Xylophone</span>](environments/index.md#robodojo)
[![deposit_coin](assets/envs/robodojo/deposit_coin.gif){ loading=lazy width="120" height="120" }<span>deposit coin</span>](environments/index.md#robodojo)
[![insert_key](assets/envs/robodojo/insert_key.gif){ loading=lazy width="120" height="120" }<span>insert key</span>](environments/index.md#robodojo)
[![build_tower](assets/envs/robodojo/build_tower.gif){ loading=lazy width="120" height="120" }<span>build tower</span>](environments/index.md#robodojo)
[![put_bottles_into_dustbin](assets/envs/robodojo/put_bottles_into_dustbin.gif){ loading=lazy width="120" height="120" }<span>put bottles into dustbin</span>](environments/index.md#robodojo)
[![play_tic_tac_toe](assets/envs/robodojo/play_tic_tac_toe.gif){ loading=lazy width="120" height="120" }<span>play tic tac toe</span>](environments/index.md#robodojo)
[![classify_objects](assets/envs/robodojo/classify_objects.gif){ loading=lazy width="120" height="120" }<span>classify objects</span>](environments/index.md#robodojo)
[![fill_pen_holder](assets/envs/robodojo/fill_pen_holder.gif){ loading=lazy width="120" height="120" }<span>fill pen holder</span>](environments/index.md#robodojo)
[![fill_egg_holder](assets/envs/robodojo/fill_egg_holder.gif){ loading=lazy width="120" height="120" }<span>fill egg holder</span>](environments/index.md#robodojo)
[![organize_table](assets/envs/robodojo/organize_table.gif){ loading=lazy width="120" height="120" }<span>organize table</span>](environments/index.md#robodojo)
[![play_stacking_toy](assets/envs/robodojo/play_stacking_toy.gif){ loading=lazy width="120" height="120" }<span>play stacking toy</span>](environments/index.md#robodojo)
[![make_kong](assets/envs/robodojo/make_kong.gif){ loading=lazy width="120" height="120" }<span>make kong</span>](environments/index.md#robodojo)
[![cover_blocks](assets/envs/robodojo/cover_blocks.gif){ loading=lazy width="120" height="120" }<span>cover blocks</span>](environments/index.md#robodojo)
[![match_and_pick_from_conveyor](assets/envs/robodojo/match_and_pick_from_conveyor.gif){ loading=lazy width="120" height="120" }<span>match and pick from conveyor</span>](environments/index.md#robodojo)
[![swap_T](assets/envs/robodojo/swap_T.gif){ loading=lazy width="120" height="120" }<span>swap T</span>](environments/index.md#robodojo)
[![press_by_number](assets/envs/robodojo/press_by_number.gif){ loading=lazy width="120" height="120" }<span>press by number</span>](environments/index.md#robodojo)
[![imitate_sorting_sequence](assets/envs/robodojo/imitate_sorting_sequence.gif){ loading=lazy width="120" height="120" }<span>imitate sorting sequence</span>](environments/index.md#robodojo)
[![swap_blocks](assets/envs/robodojo/swap_blocks.gif){ loading=lazy width="120" height="120" }<span>swap blocks</span>](environments/index.md#robodojo)
[![align_blocks](assets/envs/robodojo/align_blocks.gif){ loading=lazy width="120" height="120" }<span>align blocks</span>](environments/index.md#robodojo)
[![solve_equation](assets/envs/robodojo/solve_equation.gif){ loading=lazy width="120" height="120" }<span>solve equation</span>](environments/index.md#robodojo)
[![stack_blocks_by_language](assets/envs/robodojo/stack_blocks_by_language.gif){ loading=lazy width="120" height="120" }<span>stack blocks by language</span>](environments/index.md#robodojo)
[![general_pickup](assets/envs/robodojo/general_pickup.gif){ loading=lazy width="120" height="120" }<span>general pickup</span>](environments/index.md#robodojo)
[![classify_objects_by_language](assets/envs/robodojo/classify_objects_by_language.gif){ loading=lazy width="120" height="120" }<span>classify objects by language</span>](environments/index.md#robodojo)
[![pick_from_conveyor_by_image](assets/envs/robodojo/pick_from_conveyor_by_image.gif){ loading=lazy width="120" height="120" }<span>pick from conveyor by image</span>](environments/index.md#robodojo)
[![store_tools_in_toolbox](assets/envs/robodojo/store_tools_in_toolbox.gif){ loading=lazy width="120" height="120" }<span>store tools in toolbox</span>](environments/index.md#robodojo)
[![pour_by_language](assets/envs/robodojo/pour_by_language.gif){ loading=lazy width="120" height="120" }<span>pour by language</span>](environments/index.md#robodojo)

</div>


## Thirty seconds

```python
from torch_world.envs import PointMass
from torch_world.models import LeWM
from torch_world.nn import CLSEncoder
from torch_world.planning import CEM

model = LeWM(CLSEncoder(img_size=32, patch_size=8, embed_dim=64), action_dim=2, embed_dim=64)
episodes = PointMass(batch_size=64).collect(4)                 # pixels (64, 4, 3, 32, 32), actions (64, 4, 2)
loss = model(episodes["pixels"], episodes["actions"])["loss"]  # prediction + SIGReg

model.eval()
z, goal = model.encode(episodes["pixels"][:, :1]), model.encode(episodes["pixels"][:, -1:])[:, 0]
actions = CEM(horizon=5, action_dim=2)(model, z, goal)         # (64, 5, 2)
```

This trains LeWM, a JEPA world model, on a point mass and plans toward a goal image. Read the [quick start](getting-started/quickstart.md) for what each line does, or open the [tour notebook](examples/intro.ipynb).
