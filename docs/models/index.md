# Models

Models are grouped into **families**. Models in a family share a way of learning and a base class, so they can be swapped and compared on the same data, environments and planners. The families are **JEPA**, **DINO**, **TD-MPC**, **Dreamer** and **world action models**; more families and models will be added.

<div class="grid cards tw-models tw-models--families">

-   :material-vector-link:{ .lg } **[JEPA](jepa/index.md)**

    Predict the *representation* of a target, never its pixels; with actions, a latent world model.

-   :material-school-outline:{ .lg } **[DINO](dino/index.md)**

    Self-distillation onto learned prototypes, and world models on its frozen features.

-   :material-chart-timeline-variant:{ .lg } **[TD-MPC](tdmpc/index.md)**

    Decoder-free latent dynamics trained by TD learning, planned for return.

-   :material-weather-night:{ .lg } **[Dreamer](dreamer/index.md)**

    Learn a world model by reconstructing pixels, and learn to act in its imagination.

-   :material-movie-open-play-outline:{ .lg } **[World action models](wam/index.md)**

    Denoise future video and the actions that produce it together; the policy is the model.

</div>

## Every model at a glance

| family | group | model | input | base class | notebook |
|---|---|---|---|---|---|
| JEPA | Action | [LeWM](jepa/lewm.md) | frames + actions | `WorldModel` | [action/lewm](../examples/action/lewm.ipynb) |
| JEPA | Action | [PLDM](jepa/pldm.md) | frames + actions | `WorldModel` | [action/pldm](../examples/action/pldm.ipynb) |
| JEPA | Action | [LpWM](jepa/lpwm.md) | frames + actions | `WorldModel` | [action/lpwm](../examples/action/lpwm.ipynb) |
| JEPA | Action | [Flow-JEPA](jepa/flowjepa.md) | frames + actions | `WorldModel` | [action/flowjepa](../examples/action/flowjepa.ipynb) |
| JEPA | Action | [Variational JEPA](jepa/varjepa.md) | frames + actions | `WorldModel` | [action/varjepa](../examples/action/varjepa.ipynb) |
| JEPA | Action | [AdaJEPA](jepa/adajepa.md) | frames + actions | `WorldModel` | [action/adajepa](../examples/action/adajepa.ipynb) |
| JEPA | Action | [Causal-JEPA](jepa/causaljepa.md) | object slots (frozen slot encoder) + actions | `WorldModel` | [action/causaljepa](../examples/action/causaljepa.ipynb) |
| JEPA | Action | [CF-JEPA](jepa/cfjepa.md) | frames + actions | `WorldModel` | [action/cfjepa](../examples/action/cfjepa.ipynb) |
| JEPA | Action | [TD-JEPA](jepa/tdjepa.md) | transitions (frames + actions) | `JEPA` (own policy) | [action/tdjepa](../examples/action/tdjepa.ipynb) |
| JEPA | Action | [HWM](jepa/hwm.md) | frames + actions | `WorldModel` | [action/hwm](../examples/action/hwm.ipynb) |
| JEPA | Action | [H-JEPA](jepa/hjepa.md) | frames + actions | `WorldModel` | [action/hjepa](../examples/action/hjepa.ipynb) |
| JEPA | Action | [LeWAM](jepa/lewam.md) | frames + actions (+ goal image) | `WorldActionModel` | [action/lewam](../examples/action/lewam.ipynb) |
| JEPA | Action | [RoboJEPA](jepa/robojepa.md) | multi-view frames + actions + states | `WorldModel` | [action/robojepa](../examples/action/robojepa.ipynb) |
| JEPA | Vision-language-action | [VLA-JEPA](jepa/vlajepa.md) | image + instruction (+ future frames as targets) | `WorldActionModel` | [action/vlajepa](../examples/action/vlajepa.ipynb) |
| JEPA | Vision | [I-JEPA](jepa/ijepa.md) | images | `JEPA` | [vision/ijepa](../examples/vision/ijepa.ipynb) |
| JEPA | Vision | [LeJEPA](jepa/lejepa.md) | images | `JEPA` | [vision/lejepa](../examples/vision/lejepa.ipynb) |
| JEPA | Vision | [V-JEPA](jepa/vjepa.md) | video | `JEPA` | [vision/vjepa](../examples/vision/vjepa.ipynb) |
| JEPA | Vision | [V-JEPA 2.1](jepa/vjepa21.md) | video or images | `JEPA` | [vision/vjepa21](../examples/vision/vjepa21.ipynb) |
| JEPA | Vision | [LeVJEPA](jepa/levjepa.md) | video | `JEPA` | [vision/levjepa](../examples/vision/levjepa.ipynb) |
| JEPA | Vision-language | [VL-JEPA](jepa/vljepa.md) | images or video + text | `JEPA` | [vqa/vljepa](../examples/vqa/vljepa.ipynb) |
| DINO | Action | [DINO-WM](dino/dinowm.md) | frames + actions + proprioception | `WorldModel` | [action/dinowm](../examples/action/dinowm.ipynb) |
| DINO | Vision | [DINOv2](dino/dinov2.md) | images | `nn.Module` | [vision/dinov2](../examples/vision/dinov2.ipynb) |
| TD-MPC | Control | [TD-MPC2](tdmpc/tdmpc2.md) | states or frames + actions + rewards | `WorldModel` | [action/tdmpc2](../examples/action/tdmpc2.ipynb) |
| Dreamer | Action | [DreamerV3](dreamer/dreamerv3.md) | frames + actions (+ rewards) | `WorldModel` | [action/dreamerv3](../examples/action/dreamerv3.ipynb) |
| WAM | Action | [DreamZero](wam/dreamzero.md) | video-VAE latents + actions (+ states, goal image) | `WorldActionModel` | [action/dreamzero](../examples/action/dreamzero.ipynb) |

Every model's `forward` returns a dict with a scalar `"loss"`. Every `WorldModel` also has `encode`, `predict` and `rollout`, and plans with [`CEM`](../guides/planning.md) out of the box; TD-MPC2 plans for reward with `MPPI`. A `WorldActionModel` also samples its own actions (`sample_actions`) and plans with `WAMPlanner`. To add a model, see [Bring your own model](../guides/custom-model.md).
