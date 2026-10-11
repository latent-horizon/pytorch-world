# JEPA

A **joint-embedding predictive architecture** encodes a context and a target, and trains a predictor to map the context's embedding to the target's. Because the loss lives in embedding space, the model spends its capacity on what is predictable instead of on texture and noise. When the context is the present, the target is the future and the predictor reads the action taken, a JEPA becomes a **latent world model**.

All models in this family live in `torch_world.models.jepa` and are exported from `torch_world.models`.

## Images

<div class="grid cards" markdown>

-   **[I-JEPA](ijepa.md)**

    ---

    Predict the features of masked image blocks from one visible context block.

    EMA target · smooth L1 · multi-block masks

-   **[LeJEPA](lejepa.md)**

    ---

    Every view predicts the mean of the global views; SIGReg keeps embeddings Gaussian. No teacher, no stop-gradient, no schedules.

    SIGReg · multi-view

</div>

## Video

<div class="grid cards" markdown>

-   **[V-JEPA](vjepa.md)**

    ---

    Predict the features of masked spatio-temporal tubes. Covers V-JEPA 2 with 3D RoPE.

    EMA target · L1 · tube masks

</div>

## Vision-language

<div class="grid cards" markdown>

-   **[VL-JEPA](vljepa.md)**

    ---

    Predict the embedding of a text answer from visual tokens and a question; decode text only on demand.

    frozen X-encoder · InfoNCE

</div>

## World models

Action-conditioned JEPAs. They subclass `WorldModel`, so `rollout`, `cost` and [CEM planning](../../guides/planning.md) work without extra code.

<div class="grid cards" markdown>

-   **[LeWM](lewm.md)**

    ---

    End-to-end from pixels: next-embedding MSE plus SIGReg, with an adaLN-conditioned causal predictor.

    trained encoder · SIGReg

-   **[DINO-WM](dinowm.md)**

    ---

    Dynamics over the patch features of a frozen DINOv2 encoder, with proprioception.

    frozen encoder · MSE

</div>

## Robotics

World models for robot control: several cameras, proprioceptive state, and action spaces that differ between embodiments. They are `WorldModel`s too, and plan with the same `CEM`, conditioned on the robot's state and embodiment.

<div class="grid cards" markdown>

-   **[RoboJEPA](robojepa.md)**

    ---

    One predictor for many robots and cameras, in the latent space of a frozen V-JEPA 2.1 encoder.

    frozen encoder · L1 + rollout

</div>

## How they compare

| model | predicts | from | anti-collapse | loss |
|---|---|---|---|---|
| [I-JEPA](ijepa.md) | features of masked image blocks | a visible context block | EMA target | smooth L1 |
| [LeJEPA](lejepa.md) | mean embedding of global views | every view | SIGReg | MSE + SIGReg |
| [V-JEPA](vjepa.md) | features of masked video tubes | the unmasked tokens | EMA target | L1 (+ variance hinge) |
| [VL-JEPA](vljepa.md) | embedding of the target text | visual tokens + query | contrastive | InfoNCE |
| [LeWM](lewm.md) | next frame embedding | past embeddings + actions | SIGReg | MSE + SIGReg |
| [DINO-WM](dinowm.md) | next frame patch features | past features + actions | frozen encoder | MSE |
| [RoboJEPA](robojepa.md) | next frame features of every view | past features + actions + states | frozen encoder | L1, teacher forcing + rollout |

## Papers and code

| model | paper | reference code |
|---|---|---|
| `IJEPA` | [Assran et al., 2023](https://arxiv.org/abs/2301.08243) | [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) |
| `LeJEPA` | [Balestriero & LeCun, 2025](https://arxiv.org/abs/2511.08544) | [galilai-group/lejepa](https://github.com/galilai-group/lejepa) |
| `VJEPA` | [Bardes et al., 2024](https://arxiv.org/abs/2404.08471), [Assran et al., 2025](https://arxiv.org/abs/2506.09985) | [facebookresearch/jepa](https://github.com/facebookresearch/jepa), [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) |
| `VLJEPA` | [Chen et al., 2026](https://arxiv.org/abs/2512.10942) | not released |
| `LeWM` | [Maes et al., 2026](https://arxiv.org/abs/2603.19312) | [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) |
| `DINOWM` | [Zhou et al., 2025](https://arxiv.org/abs/2411.04983) | [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm) |
| `RoboJEPA` | [Zholus et al., 2026](https://arxiv.org/abs/2610.10515) | [facebookresearch/robo_jepa](https://github.com/facebookresearch/robo_jepa) |
