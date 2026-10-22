# World action models

A **world action model** (WAM) predicts the future of the world *and* the actions that produce it, with one network. A world model takes actions as input and predicts what follows; a policy maps observations to actions. A WAM does both at once, usually by denoising future video and an action chunk together, so its policy inherits what video pretraining knows about how the world moves. Recent WAMs built on pretrained video diffusion models lead open robot benchmarks.

All models in this family live in `torch_world.models.wam` and are exported from `torch_world.models`. They subclass `WorldActionModel`, a `WorldModel` with three more methods:

| method | returns |
|---|---|
| `sample_actions(z, goal)` | the next action chunk: the policy |
| `predict_future(z, actions, goal)` | the latents that chunk leads to: forward dynamics |
| `imagine(z, goal)` | both, sampled jointly |
| `decode(z)` | pixels, when the latent space has a decoder |

`WAMPlanner` turns any `WorldActionModel` into a planner: it samples proposals from the policy and ranks or refines them with the model's own dynamics. [LeWAM](../jepa/lewam.md), in the JEPA family, also implements this interface.

<div class="grid cards tw-models">

-   **[DreamZero](dreamzero.md)**

    An autoregressive joint video-action diffusion transformer on video-VAE latents, trained with chunk-wise teacher forcing.

    `flow matching` `video VAE` `autoregressive chunks`
    { .tw-chips }

</div>

## Kinds of world action models

| kind | how video and actions meet | examples |
|---|---|---|
| imagine, then execute | a video model plans, an inverse dynamics model reads actions off it | UniPi, video policies |
| video features for actions | an action head reads the features of a video model | Genie Envisioner |
| **joint video-action modeling** | one network denoises future video and actions together | **DreamZero**, UWM, UVA, Cosmos Policy |
| auxiliary video prediction | video prediction regularizes a policy | GR-1, GR-2 |

## How it differs from JEPA world models

| | JEPA world models | world action models |
|---|---|---|
| predicts | the next latent, from given actions | the next latents *and* the actions |
| latent space | learned embeddings, no decoder | a video autoencoder's latents, decodable to pixels |
| objective | regression + an anti-collapse term | flow matching (diffusion) |
| acts by | searching actions with a planner | sampling its own actions, optionally ranked by its own dynamics |

## Papers and code

| model | paper | reference code |
|---|---|---|
| `DreamZero` | [Ye et al., 2026](https://arxiv.org/abs/2602.15922) | not used |
