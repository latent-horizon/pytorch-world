# Reproduction report

How each model in the library compares with its paper. Every number on the "ours" side comes from an executed notebook or from the evaluation of a released checkpoint; nothing is estimated. Most notebooks are small, illustrative settings that train in minutes, so the comparison column says how close the setting is before any number is compared.

**Reproductions** use the paper's task, data and protocol. **Illustrations** show the method at small scale on a library environment; their numbers are not comparable with the paper's.

## Reproductions

| model | paper | ours | gap |
|---|---|---|---|
| LeWM | Push-T, goal 25 steps ahead, 50 steps: **96.0 ± 2.8%** (Maes et al., Table 5) | checkpoint [`latenthorizon/lewm-pusht`](https://huggingface.co/latenthorizon/lewm-pusht), stable-worldmodel `eval_wm.py`, MuJoCo Push-T, 500 tasks: **80.4%**; torch_world CEM, 100 tasks: **75.0%**; [notebook](../examples/action/lewm.ipynb), 100 tasks: **77%** [68, 84] | 16–21 points below |
| LeWM, reference simulator | same | checkpoint [`latenthorizon/lewm-pusht-pymunk`](https://huggingface.co/latenthorizon/lewm-pusht-pymunk) trained by torch_world on the released data, evaluated in the reference pymunk Push-T, 500 tasks: **83.8%**; the authors' checkpoint under the same evaluation: **84.2%** | matches the authors' checkpoint |
| DINO-WM | Push-T under LeWM's protocol: **92.0 ± 1.6%** (Maes et al., Table 5); under its own: **90%** (Zhou et al., Table 1) | checkpoint [`latenthorizon/dinowm-pusht`](https://huggingface.co/latenthorizon/dinowm-pusht), 50 tasks: **94.0%**; [notebook](../examples/action/dinowm.ipynb), 50 tasks: **92%** [81, 97] | matches |

The LeWM gap is the same for the authors' checkpoint: in the reference simulator, under the released evaluation code, it reaches 84.2%, not 96%. The paper's number averages three training seeds over 50 tasks; the evaluations here use one seed and 100 to 500 tasks. Planning in the MuJoCo port costs a few more points than in pymunk (80.4% against 83.8%): 98% of the expert's 25-step windows replay to the recorded state in the port, against 99% in the reference simulator.

## Illustrations

| model | paper's headline | our notebook | what it shows |
|---|---|---|---|
| I-JEPA | ImageNet linear probe, ViT-H/14: 79.3% | [vision/ijepa](../examples/vision/ijepa.ipynb): point images, position R² 0.995 (random init 0.995) | the objective and masks run; at this scale the probe does not beat a random encoder |
| V-JEPA | SSv2, frozen ViT-H: 72.2% | [vision/vjepa](../examples/vision/vjepa.ipynb): London street footage, busy-clip attentive probe 82.0% (random init 90.0%), dense motion R² 0.257 (0.435) | the objective, tube masks and EMA target on real footage; at this scale the probes do not beat a random encoder |
| LeJEPA | ImageNet linear probe, ViT-H/14: 79% | [vision/lejepa](../examples/vision/lejepa.ipynb): position R² 0.993 (random init 0.993), RankMe 1.8 | same; the low RankMe points to near-collapse |
| LeVJEPA | ViT-B attentive probes, ImageNet 50.7, SSv2 30.4 | [vision/levjepa](../examples/vision/levjepa.ipynb): London clips, region linear probe 89.8% | trains on real footage; ahead of DINOv2 on the same clips |
| DINOv2 | ImageNet linear probe, ViT-S/14: 81.1% | [vision/dinov2](../examples/vision/dinov2.ipynb): same clips, region linear probe 81.4% | the released ViT-S/14 against LeVJEPA trained here |
| VL-JEPA | GQA 61.5, POPE 85.7 (1.6B) | [vqa/vljepa](../examples/vqa/vljepa.ipynb): MedMNIST VQA by retrieval, 93.2% (3.7M trainable) | answering by predicting embeddings, on medical images |
| LeWAM | Push-T 25-step goals: 94.7% | [action/lewam](../examples/action/lewam.ipynb): ObstacleArm, policy 84/150 against 89/150 for the authors' code | parity with the reference implementation on one task |
| RoboJEPA | real Franka grasp, 8B: 67% | [action/robojepa](../examples/action/robojepa.ipynb): two simulated embodiments, 81% and 75% | one predictor across embodiments |
| TD-MPC2 | 104 online RL tasks (scores in figures) | [action/tdmpc2](../examples/action/tdmpc2.ipynb): PointMass goal reaching, 100% [94, 100] | the full TD-MPC2 update and MPPI |
| DreamZero | AgiBot G1 unseen tasks, task progress 39.5% (14B) | [action/dreamzero](../examples/action/dreamzero.ipynb): UR10e reach, 31–47% over two runs (41–59% with ranking; 22M, from scratch) | the joint video-action objective; far from the paper's scale |
| PLDM | Two-Rooms: 97.8 ± 0.7% (Sobal et al., Table 2) | [action/pldm](../examples/action/pldm.ipynb): Two-Rooms flown by the macaw: **100%** [89, 100] (greedy 38%, random 28%) | same task; CEM and one predictor instead of MPPI and an ensemble |
| V-JEPA 2.1 | SSv2 77.7%, NYUv2 depth RMSE 0.307 | [vision/vjepa21](../examples/vision/vjepa21.ipynb): same footage, attentive probe 81.8% against V-JEPA 2 86.9% (random init 90.2%), dense motion R² 0.237 against 0.281 | the context loss and deep self-supervision against V-JEPA 2; neither beats a random encoder at this scale |
| LpWM | Push-T: up to 57% above LeWM at intermediate predictor sizes | [action/lpwm](../examples/action/lpwm.ipynb): side of the wall read from which units fire, 99.7%; position R² 0.995; planning 38% | the room read from which units fire |
| Flow-JEPA | Two-Room: 100% (LeWM 87%) | [action/flowjepa](../examples/action/flowjepa.ipynb): across the wall, 66% [48, 80] | sampled futures; the paper's noise-robustness test is not included |
| Variational JEPA | toy linear system only, signal R² 0.905 at σ = 8 | [action/varjepa](../examples/action/varjepa.ipynb): variance–error correlation 0.52; planning on the mean 47%, on samples 12% | calibration of the predicted variance, and planning, which the paper does not test |
| HWM | Diverse Maze, longest goals: 83% hierarchical against 44% flat | [action/hwm](../examples/action/hwm.ipynb): across the wall, hierarchical 22% against flat 72% | hierarchical against flat CEM on the same model |
| H-JEPA | FourRoom: 81.3% (2 levels) against 40.7% (flat LeWM) | [action/hjepa](../examples/action/hjepa.ipynb): across the wall, hierarchical 97% against flat 91% | hierarchical against flat gradient planning |
| GRASP | Push-T open loop, horizon 50: 43.4% against CEM 30.2% | [planning/grasp](../examples/planning/grasp.ipynb): Two-Rooms, GRASP 22% against CEM 62% | GRASP against CEM on one LeWM |

Comparisons across models on one task are in the [benchmark results](results.md).
