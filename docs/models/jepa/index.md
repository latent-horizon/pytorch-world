# JEPA

A **joint-embedding predictive architecture** encodes a context and a target, and trains a predictor to map the context's embedding to the target's. Because the loss lives in embedding space, the model spends its capacity on what is predictable instead of on texture and noise. When the context is the present, the target is the future and the predictor reads the action taken, a JEPA becomes a **latent world model**.

All models in this family live in `torch_world.models.jepa` and are exported from `torch_world.models`.

## Action

Action-conditioned JEPAs: latent world models that predict the embedding of the next observation from past embeddings and actions, and plan toward a goal. They subclass `WorldModel`, so `rollout`, `cost` and [CEM planning](../../guides/planning.md) work without extra code; LeWAM also generates actions with its own `LeWAMPlanner`, and HWM and H-JEPA add a slower level planned with `HierarchicalPlanner`. RoboJEPA conditions on several cameras, proprioceptive state and the robot's embodiment. TD-JEPA is the exception: it predicts discounted future embeddings under a learned policy, so it subclasses `JEPA` and acts with that policy instead of a planner.

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

-   **[AdaJEPA](adajepa.md)**

    Keeps learning while it plans: after each executed action, one gradient step of the prediction loss on the last five transitions adapts the last predictor block and the encoder head.

    `test-time adaptation` `VCReg` `straightening`
    { .tw-chips }

-   **[Causal-JEPA](causaljepa.md)**

    An object-level world model over frozen slots: it hides objects' histories in training, so the predictor must infer them from the other objects and the actions.

    `frozen slots` `object masking` `Hungarian cost`
    { .tw-chips }

-   **[CF-JEPA](cfjepa.md)**

    LeWM with the latent split into controllable and uncontrollable parts: an inverse model and a gradient-reversed adversary sort the action's effects into one part, and the planner ignores the other.

    `trained encoder` `SIGReg` `distractors`
    { .tw-chips }

-   **[TD-JEPA](tdjepa.md)**

    Zero-shot RL from reward-free data: TD latent prediction of discounted future task embeddings under a family of policies; a goal image or a reward regression picks the policy.

    `trained encoder` `EMA target` `zero-shot RL`
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

## Vision-language-action

A robot policy that reads an image and an instruction, and whose latent actions also drive a JEPA world model. It subclasses `WorldActionModel`.

<div class="grid cards tw-models">

-   **[VLA-JEPA](vlajepa.md)**

    A vision-language-action policy whose VLM also drives a JEPA world model: latent-action tokens predict frozen V-JEPA 2 states of future frames, and a flow-matching expert outputs action chunks.

    `frozen target encoder` `language` `flow matching`
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
| [AdaJEPA](adajepa.md) | next latents, teacher forced | last 3 latents + actions | stop-gradient, LayerNorm, straightening, VCReg | MSE + curvature + VCReg; the same MSE at test time on recent transitions |
| [Causal-JEPA](causaljepa.md) | next slots of every object, and hidden objects' history | past object slots + action tokens | frozen object encoder | MSE on masked history + future |
| [CF-JEPA](cfjepa.md) | next embedding, as a controllable part (with actions) and an uncontrollable part (without) | past embeddings + actions | SIGReg | MSE + inverse dynamics + adversarial action loss + SIGReg |
| [TD-JEPA](tdjepa.md) | discounted future task embeddings under policy π_z (successor features) | state embedding + action + task z | EMA targets, orthonormality | TD latent prediction ×2 + orthonormality + actor |
| [HWM](hwm.md) | next latent and next waypoint | latents + actions or macro-actions | SIGReg, stop-gradient | L1, teacher forcing + rollout |
| [H-JEPA](hjepa.md) | next latent at each level | each level's latents + actions | SIGReg per level | MSE + SIGReg per level |
| [LeWAM](lewam.md) | future embeddings and an action chunk | past embeddings + actions (+ goal) | SIGReg | MSE + flow matching + SIGReg |
| [RoboJEPA](robojepa.md) | next frame features of every view | past features + actions + states | frozen encoder | L1, teacher forcing + rollout |
| [VLA-JEPA](vlajepa.md) | frozen V-JEPA 2 states of the next frames (teacher forcing), and an action chunk | current image + instruction (VLM latent-action tokens) | frozen target encoder | L1 latent prediction + action flow matching |
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
| `AdaJEPA` | [Wang et al., 2026](https://arxiv.org/abs/2606.32026) | [agentic-learning-ai-lab/adajepa](https://github.com/agentic-learning-ai-lab/adajepa) |
| `CausalJEPA` | [Nam et al., 2026](https://arxiv.org/abs/2602.11389) | [galilai-group/cjepa](https://github.com/galilai-group/cjepa) |
| `CFJEPA` | [Byrd et al., 2026](https://arxiv.org/abs/2610.00727) | not released |
| `TDJEPA` | [Bagatella et al., 2025](https://arxiv.org/abs/2510.00739) | not released |
| `HWM` | [Zhang et al., 2026](https://arxiv.org/abs/2604.03208) | [kevinghst/HWM_PLDM](https://github.com/kevinghst/HWM_PLDM) |
| `HJEPA` | [Zhang et al., 2026](https://arxiv.org/abs/2610.06805) | [kevinghst/H-JEPA](https://github.com/kevinghst/H-JEPA) |
| `LeWAM` | [Fu et al., 2026](https://arxiv.org/abs/2610.12818) | [MinghaoFu/lewam](https://github.com/MinghaoFu/lewam) |
| `RoboJEPA` | [Zholus et al., 2026](https://arxiv.org/abs/2610.10515) | [facebookresearch/robo_jepa](https://github.com/facebookresearch/robo_jepa) |
| `VLAJEPA` | [Sun et al., 2026](https://arxiv.org/abs/2602.10098) | [ginwind/VLA-JEPA](https://github.com/ginwind/VLA-JEPA) |
| `IJEPA` | [Assran et al., 2023](https://arxiv.org/abs/2301.08243) | [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) |
| `LeJEPA` | [Balestriero & LeCun, 2025](https://arxiv.org/abs/2511.08544) | [galilai-group/lejepa](https://github.com/galilai-group/lejepa) |
| `VJEPA` | [Bardes et al., 2024](https://arxiv.org/abs/2404.08471), [Assran et al., 2025](https://arxiv.org/abs/2506.09985) | [facebookresearch/jepa](https://github.com/facebookresearch/jepa), [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) |
| `VJEPA21` | [Mur-Labadia et al., 2026](https://arxiv.org/abs/2603.14482) | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) |
| `LeVJEPA` | [Kuhn et al., 2026](https://arxiv.org/abs/2608.27395) | [galilai-group/levjepa](https://github.com/galilai-group/levjepa) |
| `VLJEPA` | [Chen et al., 2026](https://arxiv.org/abs/2512.10942) | not released |
