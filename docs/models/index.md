# Models

Models are grouped into **families**. Models in a family share a way of learning and a base class, so they can be swapped and compared on the same data, environments and planners. The families are **JEPA** and **TD-MPC**; more families and models will be added.

<div class="grid cards" markdown>

-   :material-vector-link:{ .lg } **[JEPA](jepa/index.md)**

    ---

    Joint-embedding predictive architectures: predict the *representation* of a target from the representation of a context, never its pixels. Conditioned on actions, they become latent world models you can plan with.

    Action: LeWM · LeWAM · DINO-WM · RoboJEPA<br>Vision: I-JEPA · LeJEPA · V-JEPA · LeVJEPA<br>Vision-language: VL-JEPA

-   :material-chart-timeline-variant:{ .lg } **[TD-MPC](tdmpc/index.md)**

    ---

    Decoder-free world models for control: latent dynamics trained to predict rewards and values by temporal-difference learning, planned for return.

    TD-MPC2

</div>

## Every model at a glance

| family | group | model | input | base class | notebook |
|---|---|---|---|---|---|
| JEPA | Action | [LeWM](jepa/lewm.md) | frames + actions | `WorldModel` | [action/lewm](../examples/action/lewm.ipynb) |
| JEPA | Action | [LeWAM](jepa/lewam.md) | frames + actions (+ goal image) | `WorldModel` | [action/lewam](../examples/action/lewam.ipynb) |
| JEPA | Action | [DINO-WM](jepa/dinowm.md) | frames + actions + proprioception | `WorldModel` | [action/dinowm](../examples/action/dinowm.ipynb) |
| JEPA | Action | [RoboJEPA](jepa/robojepa.md) | multi-view frames + actions + states | `WorldModel` | [action/robojepa](../examples/action/robojepa.ipynb) |
| JEPA | Vision | [I-JEPA](jepa/ijepa.md) | images | `JEPA` | [vision/ijepa](../examples/vision/ijepa.ipynb) |
| JEPA | Vision | [LeJEPA](jepa/lejepa.md) | images | `JEPA` | [vision/lejepa](../examples/vision/lejepa.ipynb) |
| JEPA | Vision | [V-JEPA](jepa/vjepa.md) | video | `JEPA` | [vision/vjepa](../examples/vision/vjepa.ipynb) |
| JEPA | Vision | [LeVJEPA](jepa/levjepa.md) | video | `JEPA` | [vision/levjepa](../examples/vision/levjepa.ipynb) |
| JEPA | Vision-language | [VL-JEPA](jepa/vljepa.md) | images or video + text | `JEPA` | [vqa/vljepa](../examples/vqa/vljepa.ipynb) |
| TD-MPC | Control | [TD-MPC2](tdmpc/tdmpc2.md) | states or frames + actions + rewards | `WorldModel` | [action/tdmpc2](../examples/action/tdmpc2.ipynb) |

Every model's `forward` returns a dict with a scalar `"loss"`. Every `WorldModel` also has `encode`, `predict` and `rollout`, and plans with [`CEM`](../guides/planning.md) out of the box; TD-MPC2 plans for reward with `MPPI`. To add a model, see [Bring your own model](../guides/custom-model.md).
