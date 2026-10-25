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

![pytorch-world](../assets/banner-dark.png)

PyTorch のための世界モデル。世界がどう変化するかを学習し、次に起こることを想像してプランニングします。共通の部品により、どのモデルも短く、比較しやすく保たれています。

<div class="tw-install">

```bash
uv add torch_world
```

</div>

<div class="tw-actions">

[はじめる](getting-started/quickstart.md){ .tw-button .tw-button--primary }
[モデル](models/index.md){ .tw-button }
[API](api/index.md){ .tw-button }

</div>

</div>


<div class="grid cards">

-   :material-family-tree:{ .lg } **モデルファミリー**

    ---

    学習方法ごとにまとめたモデル群：画像の事前学習から複数ロボットの制御までをカバーする JEPA、DINO、報酬駆動の制御のための TD-MPC、そして動画を想像して行動する世界行動モデル。ファミリーは今後も追加されます。

    [:octicons-arrow-right-24: モデル](models/index.md)

-   :material-earth:{ .lg } **共通の世界モデルインターフェース**

    ---

    `encode`、`predict`、`rollout` はすべての世界モデルで共通なので、モデルを入れ替えて学習・比較・プランニングできます。

    [:octicons-arrow-right-24: コンセプト](getting-started/concepts.md)

-   :material-map-marker-path:{ .lg } **プランニング**

    ---

    モデルの想像の中で行動を探索します。クロスエントロピー法 (CEM) は追加のコードなしで任意の世界モデルに使えます。

    [:octicons-arrow-right-24: プランニングガイド](guides/planning.md)

-   :material-robot-industrial:{ .lg } **環境**

    ---

    190 の環境：プランニング用パズルから Franka によるマニピュレーションまでの 33 の組み込みタスクに加え、OGBench、DMControl、Meta-World、ManiSkill、MyoSuite、RoboCasa、RoboDojo。

    [:octicons-arrow-right-24: 環境](environments/index.md)

-   :material-database-outline:{ .lg } **データセット**

    ---

    ダイナミクス学習のための収集済み経験のウィンドウに加え、視覚言語モデル向けの医療画像と質問。

    [:octicons-arrow-right-24: データセット](datasets/index.md)

-   :material-cube-outline:{ .lg } **ビルディングブロック**

    ---

    画像・動画向けの Vision Transformer、条件付き予測器、マスキング戦略、目的関数など、すべてのファミリーで共有される部品。

    [:octicons-arrow-right-24: API リファレンス](api/index.md)

</div>

## 環境 {#environments}

190 の環境：組み込み 33、OGBench 13、DMControl 20、Meta-World 10、ManiSkill 12、MyoSuite 10、RoboCasa 50、RoboDojo 42。組み込みタスクは検証済みの成功エピソードを表示します。OGBench のタスクはデータセットのエピソードを、Meta-World のタスクはスクリプト化されたエキスパートを、DMControl・ManiSkill・MyoSuite のタスクはランダム方策を、RoboCasa のタスクは人間によるデモンストレーションを、RoboDojo のタスクは公式のタスク動画を再生します。各環境の詳細は[環境](environments/index.md)を参照してください。

<div class="tw-gallery">

[![ArrangeBreadBasket](../assets/envs/robocasa/ArrangeBreadBasket.gif){ loading=lazy width="120" height="120" }<span>Arrange Bread Basket</span>](environments/index.md#robocasa)
[![stack_bowls](../assets/envs/robodojo/stack_bowls.gif){ loading=lazy width="120" height="120" }<span>stack bowls</span>](environments/index.md#robodojo)
[![Checkpoints-v0](../assets/envs/success/checkpoints_v0.gif){ loading=lazy width="120" height="120" }<span>Checkpoints</span>](examples/envs/puzzles/checkpoints_v0.ipynb)
[![acrobot-swingup](../assets/envs/dmcontrol/acrobot-swingup.gif){ loading=lazy width="120" height="120" }<span>acrobot-swingup</span>](environments/index.md#deepmind-control-suite)
[![ArrangeTea](../assets/envs/robocasa/ArrangeTea.gif){ loading=lazy width="120" height="120" }<span>Arrange Tea</span>](environments/index.md#robocasa)
[![push_T](../assets/envs/robodojo/push_T.gif){ loading=lazy width="120" height="120" }<span>push T</span>](environments/index.md#robodojo)
[![pointmaze-medium-navigate-v0](../assets/envs/ogbench/pointmaze-medium-navigate.gif){ loading=lazy width="120" height="120" }<span>pointmaze-medium-navigate</span>](environments/index.md#ogbench)
[![PickCube-v1](../assets/envs/maniskill/PickCube-v1.gif){ loading=lazy width="120" height="120" }<span>PickCube-v1</span>](environments/index.md#maniskill)
[![IceMaze-v0](../assets/envs/success/icemaze_v0.gif){ loading=lazy width="120" height="120" }<span>IceMaze</span>](examples/envs/puzzles/icemaze_v0.ipynb)
[![reach-v3](../assets/envs/metaworld/reach-v3.gif){ loading=lazy width="120" height="120" }<span>reach-v3</span>](environments/index.md#meta-world)
[![myoFingerReachRandom-v0](../assets/envs/myosuite/myoFingerReachRandom-v0.gif){ loading=lazy width="120" height="120" }<span>myoFingerReachRandom-v0</span>](environments/index.md#myosuite)
[![BreadSelection](../assets/envs/robocasa/BreadSelection.gif){ loading=lazy width="120" height="120" }<span>Bread Selection</span>](environments/index.md#robocasa)
[![pack_objects_into_box](../assets/envs/robodojo/pack_objects_into_box.gif){ loading=lazy width="120" height="120" }<span>pack objects into box</span>](environments/index.md#robodojo)
[![CategorizeCondiments](../assets/envs/robocasa/CategorizeCondiments.gif){ loading=lazy width="120" height="120" }<span>Categorize Condiments</span>](environments/index.md#robocasa)
[![ball_in_cup-catch](../assets/envs/dmcontrol/ball_in_cup-catch.gif){ loading=lazy width="120" height="120" }<span>ball_in_cup-catch</span>](environments/index.md#deepmind-control-suite)
[![KeyDoor-v0](../assets/envs/success/keydoor_v0.gif){ loading=lazy width="120" height="120" }<span>KeyDoor</span>](examples/envs/puzzles/keydoor_v0.ipynb)
[![fold_clothes](../assets/envs/robodojo/fold_clothes.gif){ loading=lazy width="120" height="120" }<span>fold clothes</span>](environments/index.md#robodojo)
[![CloseBlenderLid](../assets/envs/robocasa/CloseBlenderLid.gif){ loading=lazy width="120" height="120" }<span>Close Blender Lid</span>](environments/index.md#robocasa)
[![hang_mugs](../assets/envs/robodojo/hang_mugs.gif){ loading=lazy width="120" height="120" }<span>hang mugs</span>](environments/index.md#robodojo)
[![Maze-v0](../assets/envs/success/maze_v0.gif){ loading=lazy width="120" height="120" }<span>Maze</span>](examples/envs/puzzles/maze_v0.ipynb)
[![CloseFridge](../assets/envs/robocasa/CloseFridge.gif){ loading=lazy width="120" height="120" }<span>Close Fridge</span>](environments/index.md#robocasa)
[![antmaze-medium-navigate-v0](../assets/envs/ogbench/antmaze-medium-navigate.gif){ loading=lazy width="120" height="120" }<span>antmaze-medium-navigate</span>](environments/index.md#ogbench)
[![cartpole-balance](../assets/envs/dmcontrol/cartpole-balance.gif){ loading=lazy width="120" height="120" }<span>cartpole-balance</span>](environments/index.md#deepmind-control-suite)
[![PushCube-v1](../assets/envs/maniskill/PushCube-v1.gif){ loading=lazy width="120" height="120" }<span>PushCube-v1</span>](environments/index.md#maniskill)
[![CloseToasterOvenDoor](../assets/envs/robocasa/CloseToasterOvenDoor.gif){ loading=lazy width="120" height="120" }<span>Close Toaster Oven Door</span>](environments/index.md#robocasa)
[![sweep_blocks](../assets/envs/robodojo/sweep_blocks.gif){ loading=lazy width="120" height="120" }<span>sweep blocks</span>](environments/index.md#robodojo)
[![Sokoban-v0](../assets/envs/success/sokoban_v0.gif){ loading=lazy width="120" height="120" }<span>Sokoban</span>](examples/envs/puzzles/sokoban_v0.ipynb)
[![push-v3](../assets/envs/metaworld/push-v3.gif){ loading=lazy width="120" height="120" }<span>push-v3</span>](environments/index.md#meta-world)
[![myoFingerPoseRandom-v0](../assets/envs/myosuite/myoFingerPoseRandom-v0.gif){ loading=lazy width="120" height="120" }<span>myoFingerPoseRandom-v0</span>](environments/index.md#myosuite)
[![CoffeeSetupMug](../assets/envs/robocasa/CoffeeSetupMug.gif){ loading=lazy width="120" height="120" }<span>Coffee Setup Mug</span>](environments/index.md#robocasa)
[![pour_liquid_into_cup](../assets/envs/robodojo/pour_liquid_into_cup.gif){ loading=lazy width="120" height="120" }<span>pour liquid into cup</span>](environments/index.md#robodojo)
[![CuttingToolSelection](../assets/envs/robocasa/CuttingToolSelection.gif){ loading=lazy width="120" height="120" }<span>Cutting Tool Selection</span>](environments/index.md#robocasa)
[![BallPlate-v0](../assets/envs/success/ballplate_v0.gif){ loading=lazy width="120" height="120" }<span>BallPlate</span>](examples/envs/control/ballplate_v0.ipynb)
[![cartpole-balance_sparse](../assets/envs/dmcontrol/cartpole-balance_sparse.gif){ loading=lazy width="120" height="120" }<span>cartpole-balance_sparse</span>](environments/index.md#deepmind-control-suite)
[![make_toast](../assets/envs/robodojo/make_toast.gif){ loading=lazy width="120" height="120" }<span>make toast</span>](environments/index.md#robodojo)
[![DeliverStraw](../assets/envs/robocasa/DeliverStraw.gif){ loading=lazy width="120" height="120" }<span>Deliver Straw</span>](environments/index.md#robocasa)
[![visual-antmaze-medium-navigate-v0](../assets/envs/ogbench/visual-antmaze-medium-navigate.gif){ loading=lazy width="120" height="120" }<span>visual-antmaze-medium-navigate</span>](environments/index.md#ogbench)
[![arrange_largest_number](../assets/envs/robodojo/arrange_largest_number.gif){ loading=lazy width="120" height="120" }<span>arrange largest number</span>](environments/index.md#robodojo)
[![CartpoleSwingup-v0](../assets/envs/success/cartpoleswingup_v0.gif){ loading=lazy width="120" height="120" }<span>CartpoleSwingup</span>](examples/envs/control/cartpoleswingup_v0.ipynb)
[![PullCube-v1](../assets/envs/maniskill/PullCube-v1.gif){ loading=lazy width="120" height="120" }<span>PullCube-v1</span>](environments/index.md#maniskill)
[![GarnishPancake](../assets/envs/robocasa/GarnishPancake.gif){ loading=lazy width="120" height="120" }<span>Garnish Pancake</span>](environments/index.md#robocasa)
[![cartpole-swingup](../assets/envs/dmcontrol/cartpole-swingup.gif){ loading=lazy width="120" height="120" }<span>cartpole-swingup</span>](environments/index.md#deepmind-control-suite)
[![sort_nesting_dolls_by_size](../assets/envs/robodojo/sort_nesting_dolls_by_size.gif){ loading=lazy width="120" height="120" }<span>sort nesting dolls by size</span>](environments/index.md#robodojo)
[![GatherTableware](../assets/envs/robocasa/GatherTableware.gif){ loading=lazy width="120" height="120" }<span>Gather Tableware</span>](environments/index.md#robocasa)
[![ReactionWheel-v0](../assets/envs/success/reactionwheel_v0.gif){ loading=lazy width="120" height="120" }<span>ReactionWheel</span>](examples/envs/control/reactionwheel_v0.ipynb)
[![pick-place-v3](../assets/envs/metaworld/pick-place-v3.gif){ loading=lazy width="120" height="120" }<span>pick-place-v3</span>](environments/index.md#meta-world)
[![myoElbowPose1D6MRandom-v0](../assets/envs/myosuite/myoElbowPose1D6MRandom-v0.gif){ loading=lazy width="120" height="120" }<span>myoElbowPose1D6MRandom-v0</span>](environments/index.md#myosuite)
[![GetToastedBread](../assets/envs/robocasa/GetToastedBread.gif){ loading=lazy width="120" height="120" }<span>Get Toasted Bread</span>](environments/index.md#robocasa)
[![store_laptop_and_headphones](../assets/envs/robodojo/store_laptop_and_headphones.gif){ loading=lazy width="120" height="120" }<span>store laptop and headphones</span>](environments/index.md#robodojo)
[![NewtonAcrobot-v0](../assets/envs/success/newtonacrobot_v0.gif){ loading=lazy width="120" height="120" }<span>NewtonAcrobot</span>](examples/envs/control/newtonacrobot_v0.ipynb)
[![humanoidmaze-medium-navigate-v0](../assets/envs/ogbench/humanoidmaze-medium-navigate.gif){ loading=lazy width="120" height="120" }<span>humanoidmaze-medium-navigate</span>](environments/index.md#ogbench)
[![HeatKebabSandwich](../assets/envs/robocasa/HeatKebabSandwich.gif){ loading=lazy width="120" height="120" }<span>Heat Kebab Sandwich</span>](environments/index.md#robocasa)
[![stack_blocks](../assets/envs/robodojo/stack_blocks.gif){ loading=lazy width="120" height="120" }<span>stack blocks</span>](environments/index.md#robodojo)
[![cartpole-swingup_sparse](../assets/envs/dmcontrol/cartpole-swingup_sparse.gif){ loading=lazy width="120" height="120" }<span>cartpole-swingup_sparse</span>](environments/index.md#deepmind-control-suite)
[![KettleBoiling](../assets/envs/robocasa/KettleBoiling.gif){ loading=lazy width="120" height="120" }<span>Kettle Boiling</span>](environments/index.md#robocasa)
[![PokeCube-v1](../assets/envs/maniskill/PokeCube-v1.gif){ loading=lazy width="120" height="120" }<span>PokeCube-v1</span>](environments/index.md#maniskill)
[![NewtonPendulum-v0](../assets/envs/success/newtonpendulum_v0.gif){ loading=lazy width="120" height="120" }<span>NewtonPendulum</span>](examples/envs/control/newtonpendulum_v0.ipynb)
[![fasten_screws](../assets/envs/robodojo/fasten_screws.gif){ loading=lazy width="120" height="120" }<span>fasten screws</span>](environments/index.md#robodojo)
[![LoadDishwasher](../assets/envs/robocasa/LoadDishwasher.gif){ loading=lazy width="120" height="120" }<span>Load Dishwasher</span>](environments/index.md#robocasa)
[![insert_tubes](../assets/envs/robodojo/insert_tubes.gif){ loading=lazy width="120" height="120" }<span>insert tubes</span>](environments/index.md#robodojo)
[![cheetah-run](../assets/envs/dmcontrol/cheetah-run.gif){ loading=lazy width="120" height="120" }<span>cheetah-run</span>](environments/index.md#deepmind-control-suite)
[![Macaw-v0](../assets/envs/success/macaw_v0.gif){ loading=lazy width="120" height="120" }<span>Macaw</span>](examples/envs/navigation/macaw_v0.ipynb)
[![MakeIceLemonade](../assets/envs/robocasa/MakeIceLemonade.gif){ loading=lazy width="120" height="120" }<span>Make Ice Lemonade</span>](environments/index.md#robocasa)
[![plug_in_charger](../assets/envs/robodojo/plug_in_charger.gif){ loading=lazy width="120" height="120" }<span>plug in charger</span>](environments/index.md#robodojo)
[![antsoccer-arena-navigate-v0](../assets/envs/ogbench/antsoccer-arena-navigate.gif){ loading=lazy width="120" height="120" }<span>antsoccer-arena-navigate</span>](environments/index.md#ogbench)
[![door-open-v3](../assets/envs/metaworld/door-open-v3.gif){ loading=lazy width="120" height="120" }<span>door-open-v3</span>](environments/index.md#meta-world)
[![myoHandReachRandom-v0](../assets/envs/myosuite/myoHandReachRandom-v0.gif){ loading=lazy width="120" height="120" }<span>myoHandReachRandom-v0</span>](environments/index.md#myosuite)
[![NavigateKitchen](../assets/envs/robocasa/NavigateKitchen.gif){ loading=lazy width="120" height="120" }<span>Navigate Kitchen</span>](environments/index.md#robocasa)
[![PointMass-v0](../assets/envs/success/pointmass_v0.gif){ loading=lazy width="120" height="120" }<span>PointMass</span>](examples/envs/navigation/pointmass_v0.ipynb)
[![pour_balls_into_vase](../assets/envs/robodojo/pour_balls_into_vase.gif){ loading=lazy width="120" height="120" }<span>pour balls into vase</span>](environments/index.md#robodojo)
[![OpenCabinet](../assets/envs/robocasa/OpenCabinet.gif){ loading=lazy width="120" height="120" }<span>Open Cabinet</span>](environments/index.md#robocasa)
[![finger-spin](../assets/envs/dmcontrol/finger-spin.gif){ loading=lazy width="120" height="120" }<span>finger-spin</span>](environments/index.md#deepmind-control-suite)
[![StackCube-v1](../assets/envs/maniskill/StackCube-v1.gif){ loading=lazy width="120" height="120" }<span>StackCube-v1</span>](environments/index.md#maniskill)
[![OpenDrawer](../assets/envs/robocasa/OpenDrawer.gif){ loading=lazy width="120" height="120" }<span>Open Drawer</span>](environments/index.md#robocasa)
[![DifferentialParking-v0](../assets/envs/success/differentialparking_v0.gif){ loading=lazy width="120" height="120" }<span>DifferentialParking</span>](examples/envs/navigation/differentialparking_v0.ipynb)
[![play_Xylophone](../assets/envs/robodojo/play_Xylophone.gif){ loading=lazy width="120" height="120" }<span>play Xylophone</span>](environments/index.md#robodojo)
[![OpenStandMixerHead](../assets/envs/robocasa/OpenStandMixerHead.gif){ loading=lazy width="120" height="120" }<span>Open Stand Mixer Head</span>](environments/index.md#robocasa)
[![deposit_coin](../assets/envs/robodojo/deposit_coin.gif){ loading=lazy width="120" height="120" }<span>deposit coin</span>](environments/index.md#robodojo)
[![HurdleHopper-v0](../assets/envs/success/hurdlehopper_v0.gif){ loading=lazy width="120" height="120" }<span>HurdleHopper</span>](examples/envs/navigation/hurdlehopper_v0.ipynb)
[![cube-single-play-v0](../assets/envs/ogbench/cube-single-play.gif){ loading=lazy width="120" height="120" }<span>cube-single-play</span>](environments/index.md#ogbench)
[![finger-turn_easy](../assets/envs/dmcontrol/finger-turn_easy.gif){ loading=lazy width="120" height="120" }<span>finger-turn_easy</span>](environments/index.md#deepmind-control-suite)
[![PackIdenticalLunches](../assets/envs/robocasa/PackIdenticalLunches.gif){ loading=lazy width="120" height="120" }<span>Pack Identical Lunches</span>](environments/index.md#robocasa)
[![insert_key](../assets/envs/robodojo/insert_key.gif){ loading=lazy width="120" height="120" }<span>insert key</span>](environments/index.md#robodojo)
[![drawer-open-v3](../assets/envs/metaworld/drawer-open-v3.gif){ loading=lazy width="120" height="120" }<span>drawer-open-v3</span>](environments/index.md#meta-world)
[![myoHandPoseRandom-v0](../assets/envs/myosuite/myoHandPoseRandom-v0.gif){ loading=lazy width="120" height="120" }<span>myoHandPoseRandom-v0</span>](environments/index.md#myosuite)
[![PanTransfer](../assets/envs/robocasa/PanTransfer.gif){ loading=lazy width="120" height="120" }<span>Pan Transfer</span>](environments/index.md#robocasa)
[![PayloadTransport-v0](../assets/envs/success/payloadtransport_v0.gif){ loading=lazy width="120" height="120" }<span>PayloadTransport</span>](examples/envs/navigation/payloadtransport_v0.ipynb)
[![PlaceSphere-v1](../assets/envs/maniskill/PlaceSphere-v1.gif){ loading=lazy width="120" height="120" }<span>PlaceSphere-v1</span>](environments/index.md#maniskill)
[![build_tower](../assets/envs/robodojo/build_tower.gif){ loading=lazy width="120" height="120" }<span>build tower</span>](environments/index.md#robodojo)
[![PickPlaceCounterToCabinet](../assets/envs/robocasa/PickPlaceCounterToCabinet.gif){ loading=lazy width="120" height="120" }<span>Pick Place Counter To Cabinet</span>](environments/index.md#robocasa)
[![finger-turn_hard](../assets/envs/dmcontrol/finger-turn_hard.gif){ loading=lazy width="120" height="120" }<span>finger-turn_hard</span>](environments/index.md#deepmind-control-suite)
[![PlanarQuadrotor-v0](../assets/envs/success/planarquadrotor_v0.gif){ loading=lazy width="120" height="120" }<span>PlanarQuadrotor</span>](examples/envs/navigation/planarquadrotor_v0.ipynb)
[![put_bottles_into_dustbin](../assets/envs/robodojo/put_bottles_into_dustbin.gif){ loading=lazy width="120" height="120" }<span>put bottles into dustbin</span>](environments/index.md#robodojo)
[![PickPlaceCounterToStove](../assets/envs/robocasa/PickPlaceCounterToStove.gif){ loading=lazy width="120" height="120" }<span>Pick Place Counter To Stove</span>](environments/index.md#robocasa)
[![visual-cube-single-play-v0](../assets/envs/ogbench/visual-cube-single-play.gif){ loading=lazy width="120" height="120" }<span>visual-cube-single-play</span>](environments/index.md#ogbench)
[![PickPlaceDrawerToCounter](../assets/envs/robocasa/PickPlaceDrawerToCounter.gif){ loading=lazy width="120" height="120" }<span>Pick Place Drawer To Counter</span>](environments/index.md#robocasa)
[![play_tic_tac_toe](../assets/envs/robodojo/play_tic_tac_toe.gif){ loading=lazy width="120" height="120" }<span>play tic tac toe</span>](environments/index.md#robodojo)
[![BlockStack-v0](../assets/envs/success/blockstack_v0.gif){ loading=lazy width="120" height="120" }<span>BlockStack</span>](examples/envs/manipulation/blockstack_v0.ipynb)
[![hopper-hop](../assets/envs/dmcontrol/hopper-hop.gif){ loading=lazy width="120" height="120" }<span>hopper-hop</span>](environments/index.md#deepmind-control-suite)
[![PickPlaceSinkToCounter](../assets/envs/robocasa/PickPlaceSinkToCounter.gif){ loading=lazy width="120" height="120" }<span>Pick Place Sink To Counter</span>](environments/index.md#robocasa)
[![classify_objects](../assets/envs/robodojo/classify_objects.gif){ loading=lazy width="120" height="120" }<span>classify objects</span>](environments/index.md#robodojo)
[![RollBall-v1](../assets/envs/maniskill/RollBall-v1.gif){ loading=lazy width="120" height="120" }<span>RollBall-v1</span>](environments/index.md#maniskill)
[![BoxGate-v0](../assets/envs/success/boxgate_v0.gif){ loading=lazy width="120" height="120" }<span>BoxGate</span>](examples/envs/manipulation/boxgate_v0.ipynb)
[![drawer-close-v3](../assets/envs/metaworld/drawer-close-v3.gif){ loading=lazy width="120" height="120" }<span>drawer-close-v3</span>](environments/index.md#meta-world)
[![myoHandKeyTurnRandom-v0](../assets/envs/myosuite/myoHandKeyTurnRandom-v0.gif){ loading=lazy width="120" height="120" }<span>myoHandKeyTurnRandom-v0</span>](environments/index.md#myosuite)
[![PickPlaceToasterToCounter](../assets/envs/robocasa/PickPlaceToasterToCounter.gif){ loading=lazy width="120" height="120" }<span>Pick Place Toaster To Counter</span>](environments/index.md#robocasa)
[![fill_pen_holder](../assets/envs/robodojo/fill_pen_holder.gif){ loading=lazy width="120" height="120" }<span>fill pen holder</span>](environments/index.md#robodojo)
[![PortionHotDogs](../assets/envs/robocasa/PortionHotDogs.gif){ loading=lazy width="120" height="120" }<span>Portion Hot Dogs</span>](environments/index.md#robocasa)
[![hopper-stand](../assets/envs/dmcontrol/hopper-stand.gif){ loading=lazy width="120" height="120" }<span>hopper-stand</span>](environments/index.md#deepmind-control-suite)
[![cube-double-play-v0](../assets/envs/ogbench/cube-double-play.gif){ loading=lazy width="120" height="120" }<span>cube-double-play</span>](environments/index.md#ogbench)
[![DoorOpen-v0](../assets/envs/success/dooropen_v0.gif){ loading=lazy width="120" height="120" }<span>DoorOpen</span>](examples/envs/manipulation/dooropen_v0.ipynb)
[![fill_egg_holder](../assets/envs/robodojo/fill_egg_holder.gif){ loading=lazy width="120" height="120" }<span>fill egg holder</span>](environments/index.md#robodojo)
[![PreSoakPan](../assets/envs/robocasa/PreSoakPan.gif){ loading=lazy width="120" height="120" }<span>Pre Soak Pan</span>](environments/index.md#robocasa)
[![organize_table](../assets/envs/robodojo/organize_table.gif){ loading=lazy width="120" height="120" }<span>organize table</span>](environments/index.md#robodojo)
[![DrawerOpen-v0](../assets/envs/success/draweropen_v0.gif){ loading=lazy width="120" height="120" }<span>DrawerOpen</span>](examples/envs/manipulation/draweropen_v0.ipynb)
[![PrepareCoffee](../assets/envs/robocasa/PrepareCoffee.gif){ loading=lazy width="120" height="120" }<span>Prepare Coffee</span>](environments/index.md#robocasa)
[![pendulum-swingup](../assets/envs/dmcontrol/pendulum-swingup.gif){ loading=lazy width="120" height="120" }<span>pendulum-swingup</span>](environments/index.md#deepmind-control-suite)
[![LiftPegUpright-v1](../assets/envs/maniskill/LiftPegUpright-v1.gif){ loading=lazy width="120" height="120" }<span>LiftPegUpright-v1</span>](environments/index.md#maniskill)
[![RecycleBottlesByType](../assets/envs/robocasa/RecycleBottlesByType.gif){ loading=lazy width="120" height="120" }<span>Recycle Bottles By Type</span>](environments/index.md#robocasa)
[![play_stacking_toy](../assets/envs/robodojo/play_stacking_toy.gif){ loading=lazy width="120" height="120" }<span>play stacking toy</span>](environments/index.md#robodojo)
[![ObstacleArm-v0](../assets/envs/success/obstaclearm_v0.gif){ loading=lazy width="120" height="120" }<span>ObstacleArm</span>](examples/envs/manipulation/obstaclearm_v0.ipynb)
[![button-press-topdown-v3](../assets/envs/metaworld/button-press-topdown-v3.gif){ loading=lazy width="120" height="120" }<span>button-press-topdown-v3</span>](environments/index.md#meta-world)
[![myoHandObjHoldRandom-v0](../assets/envs/myosuite/myoHandObjHoldRandom-v0.gif){ loading=lazy width="120" height="120" }<span>myoHandObjHoldRandom-v0</span>](environments/index.md#myosuite)
[![RinseSinkBasin](../assets/envs/robocasa/RinseSinkBasin.gif){ loading=lazy width="120" height="120" }<span>Rinse Sink Basin</span>](environments/index.md#robocasa)
[![visual-cube-double-play-v0](../assets/envs/ogbench/visual-cube-double-play.gif){ loading=lazy width="120" height="120" }<span>visual-cube-double-play</span>](environments/index.md#ogbench)
[![make_kong](../assets/envs/robodojo/make_kong.gif){ loading=lazy width="120" height="120" }<span>make kong</span>](environments/index.md#robodojo)
[![ScrubCuttingBoard](../assets/envs/robocasa/ScrubCuttingBoard.gif){ loading=lazy width="120" height="120" }<span>Scrub Cutting Board</span>](environments/index.md#robocasa)
[![PegInsertion-v0](../assets/envs/success/peginsertion_v0.gif){ loading=lazy width="120" height="120" }<span>PegInsertion</span>](examples/envs/manipulation/peginsertion_v0.ipynb)
[![quadruped-run](../assets/envs/dmcontrol/quadruped-run.gif){ loading=lazy width="120" height="120" }<span>quadruped-run</span>](environments/index.md#deepmind-control-suite)
[![cover_blocks](../assets/envs/robodojo/cover_blocks.gif){ loading=lazy width="120" height="120" }<span>cover blocks</span>](environments/index.md#robodojo)
[![SearingMeat](../assets/envs/robocasa/SearingMeat.gif){ loading=lazy width="120" height="120" }<span>Searing Meat</span>](environments/index.md#robocasa)
[![match_and_pick_from_conveyor](../assets/envs/robodojo/match_and_pick_from_conveyor.gif){ loading=lazy width="120" height="120" }<span>match and pick from conveyor</span>](environments/index.md#robodojo)
[![PickPlace-v0](../assets/envs/success/pickplace_v0.gif){ loading=lazy width="120" height="120" }<span>PickPlace</span>](examples/envs/manipulation/pickplace_v0.ipynb)
[![PullCubeTool-v1](../assets/envs/maniskill/PullCubeTool-v1.gif){ loading=lazy width="120" height="120" }<span>PullCubeTool-v1</span>](environments/index.md#maniskill)
[![SeparateFreezerRack](../assets/envs/robocasa/SeparateFreezerRack.gif){ loading=lazy width="120" height="120" }<span>Separate Freezer Rack</span>](environments/index.md#robocasa)
[![quadruped-walk](../assets/envs/dmcontrol/quadruped-walk.gif){ loading=lazy width="120" height="120" }<span>quadruped-walk</span>](environments/index.md#deepmind-control-suite)
[![swap_T](../assets/envs/robodojo/swap_T.gif){ loading=lazy width="120" height="120" }<span>swap T</span>](environments/index.md#robodojo)
[![SetUpCuttingStation](../assets/envs/robocasa/SetUpCuttingStation.gif){ loading=lazy width="120" height="120" }<span>Set Up Cutting Station</span>](environments/index.md#robocasa)
[![scene-play-v0](../assets/envs/ogbench/scene-play.gif){ loading=lazy width="120" height="120" }<span>scene-play</span>](environments/index.md#ogbench)
[![PuckPush-v0](../assets/envs/success/puckpush_v0.gif){ loading=lazy width="120" height="120" }<span>PuckPush</span>](examples/envs/manipulation/puckpush_v0.ipynb)
[![peg-insert-side-v3](../assets/envs/metaworld/peg-insert-side-v3.gif){ loading=lazy width="120" height="120" }<span>peg-insert-side-v3</span>](environments/index.md#meta-world)
[![myoHandPenTwirlRandom-v0](../assets/envs/myosuite/myoHandPenTwirlRandom-v0.gif){ loading=lazy width="120" height="120" }<span>myoHandPenTwirlRandom-v0</span>](environments/index.md#myosuite)
[![SlideDishwasherRack](../assets/envs/robocasa/SlideDishwasherRack.gif){ loading=lazy width="120" height="120" }<span>Slide Dishwasher Rack</span>](environments/index.md#robocasa)
[![press_by_number](../assets/envs/robodojo/press_by_number.gif){ loading=lazy width="120" height="120" }<span>press by number</span>](environments/index.md#robodojo)
[![PuckSlide-v0](../assets/envs/success/puckslide_v0.gif){ loading=lazy width="120" height="120" }<span>PuckSlide</span>](examples/envs/manipulation/puckslide_v0.ipynb)
[![StackBowlsCabinet](../assets/envs/robocasa/StackBowlsCabinet.gif){ loading=lazy width="120" height="120" }<span>Stack Bowls Cabinet</span>](environments/index.md#robocasa)
[![imitate_sorting_sequence](../assets/envs/robodojo/imitate_sorting_sequence.gif){ loading=lazy width="120" height="120" }<span>imitate sorting sequence</span>](environments/index.md#robodojo)
[![reacher-easy](../assets/envs/dmcontrol/reacher-easy.gif){ loading=lazy width="120" height="120" }<span>reacher-easy</span>](environments/index.md#deepmind-control-suite)
[![SteamInMicrowave](../assets/envs/robocasa/SteamInMicrowave.gif){ loading=lazy width="120" height="120" }<span>Steam In Microwave</span>](environments/index.md#robocasa)
[![PushT-v1](../assets/envs/maniskill/PushT-v1.gif){ loading=lazy width="120" height="120" }<span>PushT-v1</span>](environments/index.md#maniskill)
[![PushT-v0](../assets/envs/success/pusht_v0.gif){ loading=lazy width="120" height="120" }<span>PushT</span>](examples/envs/manipulation/pusht_v0.ipynb)
[![swap_blocks](../assets/envs/robodojo/swap_blocks.gif){ loading=lazy width="120" height="120" }<span>swap blocks</span>](environments/index.md#robodojo)
[![visual-scene-play-v0](../assets/envs/ogbench/visual-scene-play.gif){ loading=lazy width="120" height="120" }<span>visual-scene-play</span>](environments/index.md#ogbench)
[![StirVegetables](../assets/envs/robocasa/StirVegetables.gif){ loading=lazy width="120" height="120" }<span>Stir Vegetables</span>](environments/index.md#robocasa)
[![align_blocks](../assets/envs/robodojo/align_blocks.gif){ loading=lazy width="120" height="120" }<span>align blocks</span>](environments/index.md#robodojo)
[![reacher-hard](../assets/envs/dmcontrol/reacher-hard.gif){ loading=lazy width="120" height="120" }<span>reacher-hard</span>](environments/index.md#deepmind-control-suite)
[![RobotCube-v0](../assets/envs/success/robotcube_v0.gif){ loading=lazy width="120" height="120" }<span>RobotCube</span>](examples/envs/manipulation/robotcube_v0.ipynb)
[![StoreLeftoversInBowl](../assets/envs/robocasa/StoreLeftoversInBowl.gif){ loading=lazy width="120" height="120" }<span>Store Leftovers In Bowl</span>](environments/index.md#robocasa)
[![solve_equation](../assets/envs/robodojo/solve_equation.gif){ loading=lazy width="120" height="120" }<span>solve equation</span>](environments/index.md#robodojo)
[![window-open-v3](../assets/envs/metaworld/window-open-v3.gif){ loading=lazy width="120" height="120" }<span>window-open-v3</span>](environments/index.md#meta-world)
[![myoHandReorient8-v0](../assets/envs/myosuite/myoHandReorient8-v0.gif){ loading=lazy width="120" height="120" }<span>myoHandReorient8-v0</span>](environments/index.md#myosuite)
[![TurnOffStove](../assets/envs/robocasa/TurnOffStove.gif){ loading=lazy width="120" height="120" }<span>Turn Off Stove</span>](environments/index.md#robocasa)
[![RobotCubeSO101-v0](../assets/envs/success/robotcubeso101_v0.gif){ loading=lazy width="120" height="120" }<span>RobotCubeSO101</span>](examples/envs/manipulation/robotcubeso101_v0.ipynb)
[![stack_blocks_by_language](../assets/envs/robodojo/stack_blocks_by_language.gif){ loading=lazy width="120" height="120" }<span>stack blocks by language</span>](environments/index.md#robodojo)
[![TurnOnElectricKettle](../assets/envs/robocasa/TurnOnElectricKettle.gif){ loading=lazy width="120" height="120" }<span>Turn On Electric Kettle</span>](environments/index.md#robocasa)
[![walker-run](../assets/envs/dmcontrol/walker-run.gif){ loading=lazy width="120" height="120" }<span>walker-run</span>](environments/index.md#deepmind-control-suite)
[![PegInsertionSide-v1](../assets/envs/maniskill/PegInsertionSide-v1.gif){ loading=lazy width="120" height="120" }<span>PegInsertionSide-v1</span>](environments/index.md#maniskill)
[![puzzle-3x3-play-v0](../assets/envs/ogbench/puzzle-3x3-play.gif){ loading=lazy width="120" height="120" }<span>puzzle-3x3-play</span>](environments/index.md#ogbench)
[![TurnOnMicrowave](../assets/envs/robocasa/TurnOnMicrowave.gif){ loading=lazy width="120" height="120" }<span>Turn On Microwave</span>](environments/index.md#robocasa)
[![SpatialArm-v0](../assets/envs/success/spatialarm_v0.gif){ loading=lazy width="120" height="120" }<span>SpatialArm</span>](examples/envs/manipulation/spatialarm_v0.ipynb)
[![general_pickup](../assets/envs/robodojo/general_pickup.gif){ loading=lazy width="120" height="120" }<span>general pickup</span>](environments/index.md#robodojo)
[![TurnOnSinkFaucet](../assets/envs/robocasa/TurnOnSinkFaucet.gif){ loading=lazy width="120" height="120" }<span>Turn On Sink Faucet</span>](environments/index.md#robocasa)
[![classify_objects_by_language](../assets/envs/robodojo/classify_objects_by_language.gif){ loading=lazy width="120" height="120" }<span>classify objects by language</span>](environments/index.md#robodojo)
[![FrankaCubeLift-v0](../assets/envs/success/frankacubelift_v0.gif){ loading=lazy width="120" height="120" }<span>FrankaCubeLift</span>](examples/envs/manipulation/frankacubelift_v0.ipynb)
[![walker-stand](../assets/envs/dmcontrol/walker-stand.gif){ loading=lazy width="120" height="120" }<span>walker-stand</span>](environments/index.md#deepmind-control-suite)
[![WaffleReheat](../assets/envs/robocasa/WaffleReheat.gif){ loading=lazy width="120" height="120" }<span>Waffle Reheat</span>](environments/index.md#robocasa)
[![pick_from_conveyor_by_image](../assets/envs/robodojo/pick_from_conveyor_by_image.gif){ loading=lazy width="120" height="120" }<span>pick from conveyor by image</span>](environments/index.md#robodojo)
[![window-close-v3](../assets/envs/metaworld/window-close-v3.gif){ loading=lazy width="120" height="120" }<span>window-close-v3</span>](environments/index.md#meta-world)
[![myoLegWalk-v0](../assets/envs/myosuite/myoLegWalk-v0.gif){ loading=lazy width="120" height="120" }<span>myoLegWalk-v0</span>](environments/index.md#myosuite)
[![WashFruitColander](../assets/envs/robocasa/WashFruitColander.gif){ loading=lazy width="120" height="120" }<span>Wash Fruit Colander</span>](environments/index.md#robocasa)
[![FrankaObstacleReach-v0](../assets/envs/success/frankaobstaclereach_v0.gif){ loading=lazy width="120" height="120" }<span>FrankaObstacleReach</span>](examples/envs/manipulation/frankaobstaclereach_v0.ipynb)
[![PlugCharger-v1](../assets/envs/maniskill/PlugCharger-v1.gif){ loading=lazy width="120" height="120" }<span>PlugCharger-v1</span>](environments/index.md#maniskill)
[![visual-puzzle-3x3-play-v0](../assets/envs/ogbench/visual-puzzle-3x3-play.gif){ loading=lazy width="120" height="120" }<span>visual-puzzle-3x3-play</span>](environments/index.md#ogbench)
[![store_tools_in_toolbox](../assets/envs/robodojo/store_tools_in_toolbox.gif){ loading=lazy width="120" height="120" }<span>store tools in toolbox</span>](environments/index.md#robodojo)
[![WashLettuce](../assets/envs/robocasa/WashLettuce.gif){ loading=lazy width="120" height="120" }<span>Wash Lettuce</span>](environments/index.md#robocasa)
[![walker-walk](../assets/envs/dmcontrol/walker-walk.gif){ loading=lazy width="120" height="120" }<span>walker-walk</span>](environments/index.md#deepmind-control-suite)
[![FrankaTablePush-v0](../assets/envs/success/frankatablepush_v0.gif){ loading=lazy width="120" height="120" }<span>FrankaTablePush</span>](examples/envs/manipulation/frankatablepush_v0.ipynb)
[![pour_by_language](../assets/envs/robodojo/pour_by_language.gif){ loading=lazy width="120" height="120" }<span>pour by language</span>](environments/index.md#robodojo)
[![WeighIngredients](../assets/envs/robocasa/WeighIngredients.gif){ loading=lazy width="120" height="120" }<span>Weigh Ingredients</span>](environments/index.md#robocasa)

</div>


## 30 秒で {#thirty-seconds}

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

これは JEPA 世界モデルである LeWM を点質量 (point mass) 環境で学習し、ゴール画像に向けてプランニングします。各行の意味は[クイックスタート](getting-started/quickstart.md)を読むか、[ツアーノートブック](examples/intro.ipynb)を開いてください。


## 1 分で {#in-one-minute}

<div class="tw-film">
<video src="../assets/video/pytorch-world.mp4" poster="../assets/video/pytorch-world-poster.jpg" title="1 分でわかる pytorch-world" preload="none" playsinline controls></video>
</div>
