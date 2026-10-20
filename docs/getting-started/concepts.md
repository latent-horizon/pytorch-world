# Concepts

## What a world model is

A **world model** is a learned model of how an environment changes. An agent acts, its actions change a hidden **state** of the world, and it receives partial **observations** of that state. A world model captures this loop: from what has been observed and an action, it predicts what comes next. An agent that can predict the consequences of its actions can rehearse them before taking them: it can plan, learn a behavior inside its own imagination, or simulate rare events instead of waiting for them.

The idea is old. Kenneth Craik suggested in 1943 that minds reason with "small-scale models" of reality, Schmidhuber trained recurrent networks to predict their own future inputs in 1990, and Ha and Schmidhuber's *World Models* (2018) trained agents inside a dream of their own making. Since then the field has split along two questions: **what the model outputs**, and **what it is trained to predict**.

## What a world model outputs

Different consumers need different projections of the same agent–world loop. World Labs' [functional taxonomy](https://www.worldlabs.ai/blog/taxonomy-of-world-models) sorts world models by output:

| kind | outputs | judged by | examples |
|---|---|---|---|
| **renderer** | observations: pixels for a viewer, often in real time from user input | visual fidelity | Genie 3, interactive video generators |
| **simulator** | state: geometry, physics and dynamics a program or a person can query | structural and physical accuracy | 3D scene generators, physics-engine digital twins |
| **planner** | actions: what to do next, given an observation and a goal | whether the agent reaches the goal | model-based control, vision-language-action models, world action models |

The kinds share what they must know about geometry, physics and dynamics, and they are converging: renderers become action-conditioned, and planners use pretrained video renderers as their backbone.

`torch_world` is about **planners**: every world model here exists to choose actions. Internally, each still renders or simulates in its own way, and the families differ in what they predict.

## What a world model predicts

| approach | predicts | trained by | in `torch_world` | elsewhere |
|---|---|---|---|---|
| **latent prediction** | the *embedding* of the next observation | prediction error in embedding space, plus a term against collapse | [JEPA](../models/jepa/index.md) (LeWM, RoboJEPA), [DINO-WM](../models/dino/dinowm.md) | V-JEPA 2-AC |
| **value-equivalent latents** | the next latent, the reward and the value of what follows | latent consistency and temporal-difference learning, no decoder | [TD-MPC2](../models/tdmpc/tdmpc2.md) | MuZero |
| **dynamics and actions together** | future latents *and* the action chunk that produces them | latent prediction plus flow matching on actions | [LeWAM](../models/jepa/lewam.md) | world action models on pretrained video diffusion |
| **reconstruction** | the next latent state, and pixels from it | likelihood of the observations | | PlaNet, Dreamer |

Predicting pixels makes the model account for everything in the frame, including texture and noise that do not matter for control; predicting only embeddings or values lets it spend capacity on what is predictable and relevant, at the cost of needing another way to keep the representation from collapsing. Models that generate video pay for pixels but inherit what large video datasets know about how the world moves.

The families also include **representation learners** that are not world models on their own, such as I-JEPA, V-JEPA, LeJEPA and [DINOv2](../models/dino/dinov2.md). They pretrain the encoders that world models build on: DINO-WM plans on frozen DINOv2 features.

## Predict representations, not pixels

The latent approach is the oldest family in `torch_world`. A **joint-embedding predictive architecture** (JEPA) encodes a *context* and a *target* and trains a predictor to map the context's embedding to the target's. It never reconstructs pixels, so capacity goes to what is predictable rather than to texture and noise.

The hard part is collapse: a constant embedding predicts itself perfectly. Models in `torch_world` avoid it in one of these ways:

| strategy | how | models |
|---|---|---|
| EMA teacher | targets come from a slowly updated copy of the encoder, with stop-gradient | I-JEPA, V-JEPA, DINOv2 |
| distribution regularizer | [SIGReg](../api/losses.md) pushes embeddings toward an isotropic Gaussian | LeJEPA, LeVJEPA, LeWM, LeWAM |
| frozen encoder | a pre-trained encoder is not trained at all | DINO-WM, RoboJEPA |
| contrastive | [InfoNCE](../api/losses.md) pushes non-matching pairs apart | VL-JEPA |
| reward and value | the latent must predict reward and value, which a constant cannot | TD-MPC2 |

## From JEPA to world model

When the context is the present and the target is the future, and the predictor also reads the action taken, the JEPA becomes a **latent world model**: it imagines the consequence of an action in embedding space. Planning then searches for the actions whose imagined outcome lands closest to the embedding of a goal.

<figure class="tw-diagram">
<svg viewBox="0 0 820 330" role="img" aria-labelledby="twd-title twd-desc" xmlns="http://www.w3.org/2000/svg">
<title id="twd-title">A latent world model, planning toward a goal</title>
<desc id="twd-desc">The observation is encoded to a latent z_t. The predictor maps z_t and an action a_t to a predicted next latent, and rolls out over several steps. A cost compares the final predicted latent with the encoded goal image, and the planner adjusts the actions to lower it.</desc>
<defs>
<marker id="twd-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="twd-head" d="M0 1 L9 5 L0 9 z"/></marker>
<marker id="twd-arrow-accent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="twd-head-accent" d="M0 1 L9 5 L0 9 z"/></marker>
<clipPath id="twd-clip-obs"><rect x="18" y="72" width="76" height="76" rx="12"/></clipPath>
<clipPath id="twd-clip-goal"><rect x="18" y="217" width="76" height="76" rx="12"/></clipPath>
</defs>
<text class="twd-lane" x="18" y="40">imagine</text>
<text class="twd-lane" x="18" y="206">goal</text>
<image href="../assets/diagrams/macaw_observation.png" x="18" y="72" width="76" height="76" clip-path="url(#twd-clip-obs)" preserveAspectRatio="xMidYMid slice"/>
<rect class="twd-frame" x="18" y="72" width="76" height="76" rx="12"/>
<text class="twd-caption" x="56" y="170" text-anchor="middle">observation</text>
<line class="twd-line" x1="100" y1="110" x2="128" y2="110"/>
<rect class="twd-box" x="130" y="88" width="96" height="44" rx="12"/>
<text x="178" y="116" text-anchor="middle">encode</text>
<line class="twd-line" x1="226" y1="110" x2="257" y2="110"/>
<circle class="twd-z" cx="288" cy="110" r="29"/>
<text class="twd-math" x="286" y="116" text-anchor="middle">z<tspan class="twd-sub" dy="4">t</tspan></text>
<line class="twd-line" x1="317" y1="110" x2="352" y2="110"/>
<rect class="twd-box twd-box-strong" x="354" y="88" width="104" height="44" rx="12"/>
<text x="406" y="116" text-anchor="middle">predict</text>
<line class="twd-line" x1="458" y1="110" x2="489" y2="110"/>
<circle class="twd-zhat" cx="520" cy="110" r="29"/>
<text class="twd-math" x="514" y="116" text-anchor="middle">ẑ<tspan class="twd-sub" dy="4">t+1</tspan></text>
<path class="twd-line" d="M520 80 C520 34 406 34 406 86"/>
<text class="twd-caption" x="463" y="40" text-anchor="middle">rollout</text>
<line class="twd-line" x1="549" y1="110" x2="618" y2="110"/>
<text class="twd-caption twd-math" x="583" y="98" text-anchor="middle">ẑ<tspan class="twd-sub" dy="3">T</tspan></text>
<rect class="twd-box twd-box-cost" x="620" y="88" width="120" height="44" rx="12"/>
<text x="680" y="116" text-anchor="middle">cost</text>
<rect class="twd-box" x="370" y="190" width="72" height="34" rx="17"/>
<text class="twd-math" x="402" y="213" text-anchor="middle">a<tspan class="twd-sub" dy="4">t</tspan></text>
<line class="twd-line" x1="406" y1="190" x2="406" y2="134"/>
<path class="twd-line twd-plan" d="M650 134 L650 207 L444 207"/>
<text class="twd-caption twd-plan-text" x="548" y="198" text-anchor="middle">plan: lower the cost</text>
<image href="../assets/diagrams/macaw_goal.png" x="18" y="217" width="76" height="76" clip-path="url(#twd-clip-goal)" preserveAspectRatio="xMidYMid slice"/>
<rect class="twd-frame" x="18" y="217" width="76" height="76" rx="12"/>
<text class="twd-caption" x="56" y="315" text-anchor="middle">goal image</text>
<line class="twd-line" x1="100" y1="255" x2="128" y2="255"/>
<rect class="twd-box" x="130" y="233" width="96" height="44" rx="12"/>
<text x="178" y="261" text-anchor="middle">encode</text>
<line class="twd-line" x1="226" y1="255" x2="257" y2="255"/>
<circle class="twd-zgoal" cx="288" cy="255" r="29"/>
<text class="twd-math" x="282" y="261" text-anchor="middle">z<tspan class="twd-sub" dy="4">goal</tspan></text>
<path class="twd-line" d="M317 255 L710 255 L710 134"/>
</svg>
<figcaption>Encode the present, imagine the future one action at a time, and plan by searching for the actions whose imagined outcome lands closest to the goal.</figcaption>
</figure>

## Three ways to choose actions

Every world model here imagines with the same `rollout`; they differ in what the imagined future is scored against and where candidate actions come from.

| planning | scores imagined futures by | proposals from | models |
|---|---|---|---|
| goal reaching with [`CEM`](../guides/planning.md) | distance of the final latent to the goal's (`cost`) | a Gaussian, refit to the best | LeWM, DINO-WM, RoboJEPA, any `WorldModel` |
| reward with [`MPPI`](../guides/planning.md#mppi-planning-for-reward) | imagined rewards plus a learned value at the horizon | a Gaussian and the policy prior | TD-MPC2 |
| policy sampling with `LeWAMPlanner` | the model's own dynamics, toward the goal | the model's flow-matching action head | LeWAM |

## Two base classes

Every model subclasses one of two classes in [`torch_world.models`](../api/models.md).

| base class | you write | you get |
|---|---|---|
| `JEPA` | `forward(...) -> {"loss": ...}` | `make_target_encoder`, `update_target(momentum)` |
| `WorldModel` | `encode`, `predict(z, actions)`, `forward` | `rollout`, `cost`, planning with [`CEM`](../guides/planning.md) |

## Conventions

**Loss dicts.** `forward` returns a dict with a scalar `"loss"` to backpropagate, plus its named components (`"pred_loss"`, `"sigreg_loss"`, ...) for logging.

**Time is dim 1.** Episodes and latents are `(B, T, ...)`. For `WorldModel`, `predict(z, actions)` maps `z[:, t]` and `actions[:, t]` to a prediction of `z[:, t + 1]`, for every `t` at once (teacher forcing).

**Actions align with states.** In collected episodes, `actions[:, t]` is applied at `states[:, t]` and produces `states[:, t + 1]`. The last action is padding.

**History.** `WorldModel.history` is the number of past latents the predictor sees. `rollout` slides a window of that length; `None` means the full past.

**Masks are index tensors.** A mask is a `(B, K)` tensor of token indices to *keep*, and a masking strategy returns lists of them, one per block. [`apply_masks`](../api/masks.md) gathers them and stacks multiple masks on the batch dim.

**Videos are channels-first.** `VisionTransformer` takes images `(B, C, H, W)` or videos `(B, C, T, H, W)`; episodes from environments are `(B, T, C, H, W)`, so transpose dims 1 and 2 before a video encoder.

## Further reading

- World Labs, [A functional taxonomy of world models](https://www.worldlabs.ai/blog/taxonomy-of-world-models), 2026: renderers, simulators and planners.
- [World model (artificial intelligence)](https://en.wikipedia.org/wiki/World_model_(artificial_intelligence)), Wikipedia: history and approaches.
- Ha and Schmidhuber, [World Models](https://arxiv.org/abs/1803.10122), 2018.
- LeCun, [A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf), 2022: the JEPA proposal.
- Hafner et al., [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551) (PlaNet), 2019, and [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104) (DreamerV3), 2023.
- Schrittwieser et al., [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265) (MuZero), 2020.
