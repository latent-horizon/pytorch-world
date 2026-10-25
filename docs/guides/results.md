# Benchmark results

Models and planners compared on one task, with one dataset and one protocol. Each table comes from an executed notebook in [`examples/comparisons`](../examples/comparisons/main_benchmark.ipynb); rerun a notebook to reproduce its table.

## Protocol

- **Task**: [`TwoRooms`](environments.md#tworooms), PLDM's two rooms joined by a door, seen as 64×64 images. Each policy gets 60 steps to reach a goal image, from starts with the goal in the **same room** (short) and in the **other room** (long: the way goes through the door): 32 tasks of each kind in the main benchmark and the encoder comparison, 64 in the others. Each notebook gives its planners' settings.
- **Data**: 1024 episodes of 65 steps from a wandering policy that keeps its previous action with probability 0.8; no rewards and no expert. Every model trains on the same episodes.
- **Success**: coming within 0.1 of the goal at any step, as in LeWM's and PLDM's protocols, with its 95% Wilson interval. Random actions set the floor.
- **Scale**: small models (predictors of 4 blocks of width 128 to 512) trained for 5 epochs on one GPU, so these numbers compare methods at a small budget, not at the papers' scale. See the [reproduction report](reproduction.md) for each model against its paper.

## Main benchmark

[comparisons/main_benchmark](../examples/comparisons/main_benchmark.ipynb): LeWM, PLDM, DINO-WM (frozen DINOv2), V-JEPA 2-AC (frozen V-JEPA 2.1) and TD-MPC2 (hindsight goals), each with its own recipe and planner.

| model | encoder | same room | other room | training | planning, ms / step |
|---|---|---|---|---|---|
| PLDM | CNN, end to end | **100% [89, 100]** | **88% [72, 95]** | 13.8 min | 23 |
| DINO-WM | DINOv2 ViT-S/14, frozen | 97% [84, 99] | 28% [16, 45] | 7.5 min | 17,300 |
| V-JEPA 2-AC | V-JEPA 2.1 ViT-B, frozen | 88% [72, 95] | 44% [28, 61] | 25.4 min | 41,800 |
| LeWM | ViT, end to end | 78% [61, 89] | 34% [20, 52] | 5.6 min | 1,220 |
| TD-MPC2 | CNN, end to end | 19% [9, 35] | 0% [0, 11] | 3.1 min | 14 |
| random actions | | 56% [39, 72] | 3% [1, 16] | | |

32 tasks of each kind. In the same room every world model but TD-MPC2 beats random actions, which reach a nearby goal by chance half of the time. Through the door PLDM is far ahead: its open-loop multi-step objective learns where the wall stops the agent, and planning with it costs a few milliseconds a step. The frozen web encoders plan well in the same room but less well across the wall, and their larger latents make planning 15 to 35 times slower than LeWM's. TD-MPC2 is built to learn online, from its own interaction; with 5 epochs of offline data and hindsight goals its value function does not learn to reach them, so it is listed for completeness, not as a fair measure of the method.

## Frozen vs end-to-end encoders

[comparisons/frozen_vs_end_to_end](../examples/comparisons/frozen_vs_end_to_end.ipynb): frozen DINOv2 and V-JEPA 2.1 encoders under DINO-WM's predictor, a frozen LeJEPA encoder pre-trained on the frames and an end-to-end LeWM, with training time and trained parameters.

| encoder | predictor | same room | other room | trained / total parameters | training | planning, ms / step |
|---|---|---|---|---|---|---|
| DINOv2 ViT-S/14, frozen | DINO-WM | 91% [76, 97] | 25% [13, 42] | 2.5M / 24.5M | 7.5 min | 17,400 |
| V-JEPA 2.1 ViT-B, frozen | DINO-WM | 91% [76, 97] | **56% [39, 72]** | 4.8M / 91.7M | 23.2 min | 29,400 |
| LeJEPA ViT, pre-trained on the frames, frozen | LeWM | 91% [76, 97] | 22% [11, 39] | 1.5M / 2.3M | 4.5 + 6.6 min | 1,230 |
| LeWM ViT, end to end | LeWM | 69% [51, 82] | 38% [23, 55] | 2.3M / 2.3M | 6.4 min | 1,230 |

32 tasks of each kind. Every frozen encoder reaches nearby goals as well as the others; the long tasks separate them. The video encoder, V-JEPA 2.1, crosses the wall most often, at 24 times the planning cost of the small encoders. A small encoder pre-trained on the frames alone (LeJEPA) matches the web image encoder (DINOv2) at a fraction of its size. Training the encoder end to end with the dynamics (LeWM) costs some accuracy nearby but does not lose ground on the long tasks.

## Model notebooks

Each model's own notebook, with the goal on the other side of the wall, reached at any step within 40 to 60 steps. The first six fly the macaw over Two-Rooms ([`Macaw(rooms=True)`](environments.md#macaw)) on 32 tasks; the others use [`TwoRooms`](environments.md#tworooms) itself, with the setting in parentheses.

| model | planner | success | random actions |
|---|---|---|---|
| [PLDM](../examples/action/pldm.ipynb) | CEM, 20 steps, path cost | **100% [89, 100]** | 28% |
| [H-JEPA](../examples/action/hjepa.ipynb) | hierarchical, gradient | **97% [84, 99]** (flat: 91%) | 22% |
| [HWM](../examples/action/hwm.ipynb) | hierarchical CEM | 22% [11, 39] (flat: 72%) | 16% |
| [Flow-JEPA](../examples/action/flowjepa.ipynb) | CEM on sampled futures | 66% [48, 80] | 12% |
| [Variational JEPA](../examples/action/varjepa.ipynb) | CEM on the mean | 47% [31, 64] (on samples: 12%) | 3% |
| [LpWM](../examples/action/lpwm.ipynb) | CEM, 5 steps | 38% [23, 55] | 9% |
| [TD-JEPA](../examples/action/tdjepa.ipynb) | none: zero-shot policy π_z, z = ψ(goal) (64 tasks) | **98% [92, 100]** | 3% |
| [AdaJEPA](../examples/action/adajepa.ipynb) | CEM, 10 steps, one adaptation step per replan (32 tasks, 32×32) | 69% [51, 82] (frozen: 44%) | 9% |
| [DreamerV3](../examples/action/dreamerv3.ipynb) | actor trained on hindsight goals (32 tasks) | 41% [26, 58] (CEM on the model state: 16%) | 3% |
| [CF-JEPA](../examples/action/cfjepa.ipynb) | CEM on the controllable part (64 tasks, 3 drifting discs) | 16% [9, 26] (LeWM: 44%) | 5% |

The settings differ from notebook to notebook (data windows, frameskip, horizon), so this table shows what each model does with its own recipe, not a controlled comparison; the comparisons below are.

## Planners

[comparisons/planners](../examples/comparisons/planners.ipynb): CEM, MPPI and GRASP on one LeWM at horizons 3, 8 and 16, on the same-room tasks with 40 steps, with planning time per step.

| planner | horizon 3 | horizon 8 | horizon 16 | ms / step (horizon 8) |
|---|---|---|---|---|
| CEM | 61% [49, 72] | 70% [58, 80] | 62% [50, 73] | 5,830 |
| MPPI | 86% [75, 92] | **98% [92, 100]** | 89% [79, 95] | 4,700 |
| GRASP | 50% [38, 62] | 50% [38, 62] | 48% [37, 60] | 9,410 |

MPPI, which weights every sample by its cost instead of keeping a top set, beats CEM at every horizon. GRASP succeeds only on the tasks a step or two away (1.3 steps on average when it does) and stalls on the rest: its default learning rates come from the authors' LeWM, whose latents differ in scale from this smaller model's, and were not tuned here.

## Flat vs hierarchical

[comparisons/flat_vs_hierarchical](../examples/comparisons/flat_vs_hierarchical.ipynb): LeWM with flat CEM against HWM and H-JEPA planned top-down, on short and long tasks.

| model | planner | same room | other room | ms / step |
|---|---|---|---|---|
| LeWM | flat CEM, 20 steps | 58% [46, 69] | 28% [19, 40] | 29,800 |
| HWM | hierarchical CEM | 41% [29, 53] | 9% [4, 19] | 9,370 |
| H-JEPA | hierarchical, gradient | **78% [67, 86]** | **67% [55, 77]** | 10,200 |

H-JEPA, whose two levels train together end to end, more than doubles flat LeWM on the long tasks, at a third of its planning time. HWM, whose upper level learns on a frozen latent space, does worse than flat planning here, as in its own notebook; at this scale its macro-actions do not yet lead the low level around the wall.
