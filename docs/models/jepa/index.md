# JEPA

A **joint-embedding predictive architecture** encodes a context and a target, and trains a predictor to map the context's embedding to the target's. Because the loss lives in embedding space, the model spends its capacity on what is predictable instead of on texture and noise. When the context is the present, the target is the future and the predictor reads the action taken, a JEPA becomes a **latent world model**.

All models in this family live in `torch_world.models.jepa` and are exported from `torch_world.models`.

## Action

Action-conditioned JEPAs: latent world models that predict the embedding of the next observation from past embeddings and actions, and plan toward a goal. They subclass `WorldModel`, so `rollout`, `cost` and [CEM planning](../../guides/planning.md) work without extra code; LeWAM also generates actions with its own `LeWAMPlanner`, and HWM and H-JEPA add a slower level planned with `HierarchicalPlanner`. RoboJEPA conditions on several cameras, proprioceptive state and the robot's embodiment.

<div class="grid cards tw-models">

-   **[LeWM](lewm.md)**

    End-to-end from pixels: next-embedding MSE plus SIGReg, with an adaLN-conditioned causal predictor.

    `trained encoder` `SIGReg`
    { .tw-chips }

-   **[PLDM](pldm.md)**

    Reward-free latent dynamics: a CNN encoder and a GRU predictor unrolled open loop, with VICReg, smoothness and inverse-dynamics terms.

    `trained encoder` `VICReg` `multi-step`
    { .tw-chips }

-   **[LpWM](lpwm.md)**

    LeWM with sparse, non-negative latents: a ReLU link and RDMReg toward a rectified Laplace instead of SIGReg.

    `trained encoder` `RDMReg` `sparse`
    { .tw-chips }

-   **[Flow-JEPA](flowjepa.md)**

    A flow-matching predictor that samples the next five embeddings jointly: many imagined futures from one state.

    `trained encoder` `SIGReg` `flow matching`
    { .tw-chips }

-   **[Variational JEPA](varjepa.md)**

    A Gaussian predictor trained on targets sampled around an EMA encoder: its variance says how unsure it is.

    `EMA target` `NLL + KL` `SIGReg`
    { .tw-chips }

-   **[HWM](hwm.md)**

    A second level over waypoints a few steps apart, driven by latent macro-actions, for top-down planning.

    `hierarchical` `L1 + rollout`
    { .tw-chips }

-   **[H-JEPA](hjepa.md)**

    Two JEPA levels trained together end to end, each with its own latent space and SIGReg; planned top-down by gradient descent.

    `hierarchical` `SIGReg` `end to end`
    { .tw-chips }

-   **[LeWAM](lewam.md)**

    World and action model: latent dynamics and a flow-matching policy that generates action chunks, in one mixture of transformers. Plans with `LeWAMPlanner`.

    `trained encoder` `SIGReg` `flow matching`
    { .tw-chips }

-   **[RoboJEPA](robojepa.md)**

    One predictor for many robots and cameras, in the latent space of a frozen V-JEPA 2.1 encoder.

    `frozen encoder` `L1 + rollout`
    { .tw-chips }

</div>

## Vision

Self-supervised representation learning from images and video. They subclass `JEPA`; their encoders are evaluated with frozen-feature [probes](../../guides/metrics.md#probing-frozen-features).

<div class="grid cards tw-models">

-   **[I-JEPA](ijepa.md)**

    Predict the features of masked image blocks from one visible context block.

    `EMA target` `smooth L1` `multi-block masks`
    { .tw-chips }

-   **[LeJEPA](lejepa.md)**

    Every view predicts the mean of the global views; SIGReg keeps embeddings Gaussian. No teacher, no stop-gradient, no schedules.

    `SIGReg` `multi-view`
    { .tw-chips }

-   **[V-JEPA](vjepa.md)**

    Predict the features of masked spatio-temporal tubes. Covers V-JEPA 2 with 3D RoPE.

    `EMA target` `L1` `tube masks`
    { .tw-chips }

-   **[V-JEPA 2.1](vjepa21.md)**

    V-JEPA 2 with a distance-weighted context loss and deep self-supervision over four encoder levels, for dense features.

    `EMA target` `L1` `dense`
    { .tw-chips }

-   **[LeVJEPA](levjepa.md)**

    LeJEPA for video: local crops of a clip predict its global view at the [cls] token, with 95% of tokens dropped and block-causal attention.

    `SIGReg` `multi-view` `token drop`
    { .tw-chips }

</div>

## Vision-language

Predict the embedding of text from images or video and a query.

<div class="grid cards tw-models">

-   **[VL-JEPA](vljepa.md)**

    Predict the embedding of a text answer from visual tokens and a question; decode text only on demand.

    `frozen X-encoder` `InfoNCE`
    { .tw-chips }

</div>

## How they compare

| model | predicts | from | anti-collapse | loss |
|---|---|---|---|---|
| [LeWM](lewm.md) | next frame embedding | past embeddings + actions | SIGReg | MSE + SIGReg |
| [PLDM](pldm.md) | next latents, open loop | first latent + actions | VICReg, smoothness, inverse dynamics | MSE + VICReg + IDM |
| [LpWM](lpwm.md) | next frame embedding (sparse, ≥ 0) | past embeddings + actions | RDMReg | MSE + RDMReg |
| [Flow-JEPA](flowjepa.md) | the next five embeddings, sampled | current embedding + actions | SIGReg | flow matching + SIGReg |
| [Variational JEPA](varjepa.md) | a Gaussian over the next embedding | past embeddings + actions | EMA target, SIGReg | NLL + KL + SIGReg |
| [HWM](hwm.md) | next latent and next waypoint | latents + actions or macro-actions | SIGReg, stop-gradient | L1, teacher forcing + rollout |
| [H-JEPA](hjepa.md) | next latent at each level | each level's latents + actions | SIGReg per level | MSE + SIGReg per level |
| [LeWAM](lewam.md) | future embeddings and an action chunk | past embeddings + actions (+ goal) | SIGReg | MSE + flow matching + SIGReg |
| [RoboJEPA](robojepa.md) | next frame features of every view | past features + actions + states | frozen encoder | L1, teacher forcing + rollout |
| [I-JEPA](ijepa.md) | features of masked image blocks | a visible context block | EMA target | smooth L1 |
| [LeJEPA](lejepa.md) | mean embedding of global views | every view | SIGReg | MSE + SIGReg |
| [V-JEPA](vjepa.md) | features of masked video tubes | the unmasked tokens | EMA target | L1 (+ variance hinge) |
| [V-JEPA 2.1](vjepa21.md) | features of masked tubes and visible tokens, at four levels | the unmasked tokens | EMA target | L1, masked + weighted context |
| [LeVJEPA](levjepa.md) | [cls] embedding of the global view | local views of the clip | SIGReg | MSE + SIGReg |
| [VL-JEPA](vljepa.md) | embedding of the target text | visual tokens + query | contrastive | InfoNCE |

## Papers and code

| model | paper | reference code |
|---|---|---|
| `LeWM` | [Maes et al., 2026](https://arxiv.org/abs/2603.19312) | [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) |
| `PLDM` | [Sobal et al., 2025](https://arxiv.org/abs/2502.14819) | [vladisai/PLDM](https://github.com/vladisai/PLDM) |
| `LpWM` | [Kuang et al., 2026](https://arxiv.org/abs/2608.22764) | [YilunKuang/lpworldmodel](https://github.com/YilunKuang/lpworldmodel) |
| `FlowJEPA` | [Huo et al., 2026](https://arxiv.org/abs/2608.29029) | [HuoYanchen/Flow-JEPA](https://github.com/HuoYanchen/Flow-JEPA) |
| `VariationalJEPA` | [Huang, 2026](https://arxiv.org/abs/2601.14354) | [YongchaoHuang/VJEPA](https://github.com/YongchaoHuang/VJEPA) |
| `HWM` | [Zhang et al., 2026](https://arxiv.org/abs/2604.03208) | [kevinghst/HWM_PLDM](https://github.com/kevinghst/HWM_PLDM) |
| `HJEPA` | [Zhang et al., 2026](https://arxiv.org/abs/2610.06805) | [kevinghst/H-JEPA](https://github.com/kevinghst/H-JEPA) |
| `LeWAM` | [Fu et al., 2026](https://arxiv.org/abs/2610.12818) | [MinghaoFu/lewam](https://github.com/MinghaoFu/lewam) |
| `RoboJEPA` | [Zholus et al., 2026](https://arxiv.org/abs/2610.10515) | [facebookresearch/robo_jepa](https://github.com/facebookresearch/robo_jepa) |
| `IJEPA` | [Assran et al., 2023](https://arxiv.org/abs/2301.08243) | [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) |
| `LeJEPA` | [Balestriero & LeCun, 2025](https://arxiv.org/abs/2511.08544) | [galilai-group/lejepa](https://github.com/galilai-group/lejepa) |
| `VJEPA` | [Bardes et al., 2024](https://arxiv.org/abs/2404.08471), [Assran et al., 2025](https://arxiv.org/abs/2506.09985) | [facebookresearch/jepa](https://github.com/facebookresearch/jepa), [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) |
| `VJEPA21` | [Mur-Labadia et al., 2026](https://arxiv.org/abs/2603.14482) | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) |
| `LeVJEPA` | [Kuhn et al., 2026](https://arxiv.org/abs/2608.27395) | [galilai-group/levjepa](https://github.com/galilai-group/levjepa) |
| `VLJEPA` | [Chen et al., 2026](https://arxiv.org/abs/2512.10942) | not released |
