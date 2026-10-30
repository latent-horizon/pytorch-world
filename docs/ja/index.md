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

![pytorch-world](../assets/banner-light.png){ .tw-logo--light }
![pytorch-world](../assets/banner-dark.png){ .tw-logo--dark }

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

</div>

<div class="tw-more">

[すべての環境を見る :octicons-arrow-right-24:](environments/index.md)

</div>


## コントリビューター {#contributors}

pytorch-world はオープンに開発されています。コード、モデル、環境、アイデアを寄せてくださるすべての方に感謝します。

<div class="tw-contributors" data-repo="latent-horizon/pytorch-world">

[GitHub のコントリビューター](https://github.com/latent-horizon/pytorch-world/graphs/contributors)

</div>

<div class="tw-more">

[コントリビュートする :octicons-arrow-right-24:](contributing.md)

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
