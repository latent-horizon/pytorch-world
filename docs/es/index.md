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

World models para PyTorch. Aprende cómo evoluciona un mundo y luego planifica imaginando lo que viene después. Las piezas compartidas mantienen cada modelo breve y fácil de comparar.

<div class="tw-install">

```bash
uv add torch_world
```

</div>

<div class="tw-actions">

[Empezar](getting-started/quickstart.md){ .tw-button .tw-button--primary }
[Modelos](models/index.md){ .tw-button }
[API](api/index.md){ .tw-button }

</div>

</div>


<div class="grid cards">

-   :material-family-tree:{ .lg } **Familias de modelos**

    ---

    Modelos agrupados según cómo aprenden: JEPA, desde el preentrenamiento con imágenes hasta el control multirrobot; DINO; TD-MPC, para control guiado por recompensas; y world action models, que imaginan vídeo y actúan. Llegarán más familias.

    [:octicons-arrow-right-24: Modelos](models/index.md)

-   :material-earth:{ .lg } **Una sola interfaz de world model**

    ---

    `encode`, `predict` y `rollout` son iguales en todos los world models, así que pueden entrenarse, compararse y usarse para planificar de forma intercambiable.

    [:octicons-arrow-right-24: Conceptos](getting-started/concepts.md)

-   :material-map-marker-path:{ .lg } **Planificación**

    ---

    Busca acciones dentro de la imaginación de un modelo. El método de entropía cruzada funciona con cualquier world model, sin código adicional.

    [:octicons-arrow-right-24: Guía de planificación](guides/planning.md)

-   :material-robot-industrial:{ .lg } **Entornos**

    ---

    190 entornos: 33 tareas integradas, desde puzles de planificación hasta manipulación con Franka, además de OGBench, DMControl, Meta-World, ManiSkill, MyoSuite, RoboCasa y RoboDojo.

    [:octicons-arrow-right-24: Entornos](environments/index.md)

-   :material-database-outline:{ .lg } **Datasets**

    ---

    Ventanas sobre experiencia recopilada para entrenar dinámicas, además de imágenes médicas y preguntas para modelos de visión y lenguaje.

    [:octicons-arrow-right-24: Datasets](datasets/index.md)

-   :material-cube-outline:{ .lg } **Componentes básicos**

    ---

    Vision transformers para imágenes y vídeo, predictores condicionados, estrategias de enmascarado y objetivos, compartidos por todas las familias.

    [:octicons-arrow-right-24: Referencia de la API](api/index.md)

</div>

## Entornos {#environments}

190 entornos: 33 integrados, 13 de OGBench, 20 de DMControl, 10 de Meta-World, 12 de ManiSkill, 10 de MyoSuite, 50 de RoboCasa y 42 de RoboDojo. Las tareas integradas muestran un episodio exitoso verificado; las tareas de OGBench reproducen un episodio de su dataset, las de Meta-World su experto programado, las de DMControl, ManiSkill y MyoSuite una política aleatoria, las de RoboCasa una demostración humana y las de RoboDojo su vídeo oficial de la tarea. Consulta [Entornos](environments/index.md) para ver cada uno.

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

[Ver todos los entornos :octicons-arrow-right-24:](environments/index.md)

</div>


## Colaboradores {#contributors}

pytorch-world se construye en abierto. Gracias a todas las personas que aportan código, modelos, entornos e ideas.

<div class="tw-contributors" data-repo="latent-horizon/pytorch-world">

[Colaboradores en GitHub](https://github.com/latent-horizon/pytorch-world/graphs/contributors)

</div>

<div class="tw-more">

[Cómo contribuir :octicons-arrow-right-24:](contributing.md)

</div>


## Treinta segundos {#thirty-seconds}

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

Esto entrena LeWM, un world model JEPA, sobre un point mass y planifica hacia una imagen objetivo. Lee el [inicio rápido](getting-started/quickstart.md) para saber qué hace cada línea, o abre el [notebook de introducción](examples/intro.ipynb).


## En un minuto {#in-one-minute}

<div class="tw-film">
<video src="../assets/video/pytorch-world.mp4" poster="../assets/video/pytorch-world-poster.jpg" title="pytorch-world en un minuto" preload="none" playsinline controls></video>
</div>
