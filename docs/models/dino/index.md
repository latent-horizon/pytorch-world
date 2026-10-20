# DINO

**Self-distillation** learns representations with no labels and no predictor network. A student network is trained to match the outputs of a teacher, an exponential moving average of the student, on different crops of the same image. Both map their features to scores over a set of learned **prototypes**, and the student matches the teacher's sharpened, centered assignment by cross-entropy. Centering and sharpening the teacher's outputs, rather than a regularizer on the embeddings, keep the targets from collapsing.

All models in this family live in `torch_world.models.dino` and are exported from `torch_world.models`. Their `forward` returns a dict with a scalar `"loss"`.

## Vision

Self-distillation pretraining of an image encoder. Call `update_teacher(momentum)` after each optimizer step.

<div class="grid cards tw-models">

-   **[DINOv2](dinov2.md)**

    DINO on [cls] tokens and iBOT on masked patches, from one shared head, with KoLeo and a Sinkhorn-Knopp-centered EMA teacher.

    `self-distillation` `multi-crop` `masked patches`
    { .tw-chips }

</div>

## Action

World models built on the features of a frozen DINO encoder. They subclass `WorldModel`, so `rollout`, `cost` and [CEM planning](../../guides/planning.md) work without extra code.

<div class="grid cards tw-models">

-   **[DINO-WM](dinowm.md)**

    Dynamics over the patch features of a frozen DINOv2 encoder, with proprioception.

    `frozen encoder` `MSE`
    { .tw-chips }

</div>

## How it differs from JEPA

| | JEPA | DINO |
|---|---|---|
| target | the embedding of a target view | the teacher's assignment of a view to prototypes |
| loss | regression (L1, MSE) | cross-entropy between distributions |
| against collapse | EMA target, or SIGReg | EMA teacher with centering and sharpening, plus KoLeo |
| predictor | yes, for most models | none: views are matched directly |

## Papers and code

| model | paper | reference code |
|---|---|---|
| `DINOv2` | [Oquab et al., 2024](https://arxiv.org/abs/2304.07193) | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) |
| `DINOWM` | [Zhou et al., 2025](https://arxiv.org/abs/2411.04983) | [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm) |
