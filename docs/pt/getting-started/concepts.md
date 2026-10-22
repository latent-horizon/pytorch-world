# Conceitos

## O que é um world model {#what-a-world-model-is}

Um **world model** é um modelo aprendido de como um environment muda. Um agent age, suas actions mudam um **state** oculto do mundo, e ele recebe **observations** parciais desse state. Um world model captura esse ciclo: a partir do que foi observado e de uma action, ele prevê o que vem a seguir. Um agent capaz de prever as consequências de suas actions pode ensaiá-las antes de executá-las: pode planejar, aprender um comportamento dentro da própria imaginação ou simular eventos raros em vez de esperar por eles.

A ideia é antiga. Kenneth Craik sugeriu em 1943 que mentes raciocinam com "modelos em pequena escala" da realidade, Schmidhuber treinou redes recorrentes para prever suas próprias entradas futuras em 1990, e *World Models* (2018), de Ha e Schmidhuber, treinou agents dentro de um sonho criado por eles mesmos. Desde então, a área se dividiu em torno de duas perguntas: **o que o modelo produz como saída** e **o que ele é treinado para prever**.

## O que um world model produz {#what-a-world-model-outputs}

Consumidores diferentes precisam de projeções diferentes do mesmo ciclo agent–mundo. A [taxonomia funcional](https://www.worldlabs.ai/blog/taxonomy-of-world-models) da World Labs classifica world models pela saída:

| tipo | saídas | avaliado por | exemplos |
|---|---|---|---|
| **renderer** | observations: pixels para quem assiste, muitas vezes em tempo real a partir da entrada do usuário | fidelidade visual | Genie 3, geradores de vídeo interativos |
| **simulator** | state: geometria, física e dinâmica que um programa ou uma pessoa pode consultar | precisão estrutural e física | geradores de cenas 3D, digital twins baseados em physics engines |
| **planner** | actions: o que fazer a seguir, dada uma observation e um goal | se o agent alcança o goal | controle baseado em modelo, vision-language-action models, world action models |

Os tipos compartilham o que precisam saber sobre geometria, física e dinâmica, e estão convergindo: renderers passam a ser condicionados a actions, e planners usam renderers de vídeo pré-treinados como backbone.

`torch_world` trata de **planners**: todo world model aqui existe para escolher actions. Internamente, cada um ainda renderiza ou simula à sua maneira, e as famílias diferem no que preveem.

## O que um world model prevê {#what-a-world-model-predicts}

| abordagem | prevê | treinado por | em `torch_world` | em outros lugares |
|---|---|---|---|---|
| **latent prediction** | o *embedding* da próxima observation | erro de predição no embedding space, mais um termo contra o collapse | [JEPA](../models/jepa/index.md) (LeWM, RoboJEPA), [DINO-WM](../models/dino/dinowm.md) | V-JEPA 2-AC |
| **value-equivalent latents** | o próximo latent, o reward e o value do que vem depois | consistência de latents e temporal-difference learning, sem decoder | [TD-MPC2](../models/tdmpc/tdmpc2.md) | MuZero |
| **generative video and action** | frames futuros (como video-VAE latents) *e* o action chunk que os produz | denoising (flow matching) | [world action models](../models/wam/index.md) (DreamZero), [LeWAM](../models/jepa/lewam.md) | policies de robôs pré-treinadas em vídeo |
| **reconstruction** | o próximo latent state, e os pixels a partir dele | likelihood das observations | | PlaNet, Dreamer |

Prever pixels obriga o modelo a dar conta de tudo no frame, inclusive textura e ruído que não importam para o controle; prever apenas embeddings ou values permite gastar capacidade no que é previsível e relevante, ao custo de precisar de outra forma de impedir o collapse da representação. Modelos generativos pagam pelos pixels, mas herdam o que grandes datasets de vídeo sabem sobre como o mundo se move.

As famílias também incluem **representation learners** que não são world models por si só, como I-JEPA, V-JEPA, LeJEPA e [DINOv2](../models/dino/dinov2.md). Eles pré-treinam os encoders sobre os quais os world models são construídos: DINO-WM planeja sobre features congeladas do DINOv2.

## Preveja representações, não pixels {#predict-representations-not-pixels}

A abordagem latent é a família mais antiga em `torch_world`. Uma **joint-embedding predictive architecture** (JEPA) codifica um *context* e um *target* e treina um predictor para mapear o embedding do context no do target. Ela nunca reconstrói pixels, então a capacidade vai para o que é previsível, e não para textura e ruído.

A parte difícil é o collapse: um embedding constante prevê a si mesmo perfeitamente. Os modelos em `torch_world` o evitam de uma destas formas:

| estratégia | como | modelos |
|---|---|---|
| EMA teacher | os targets vêm de uma cópia do encoder atualizada lentamente, com stop-gradient | I-JEPA, V-JEPA, DINOv2 |
| distribution regularizer | [SIGReg](../api/losses.md) empurra os embeddings em direção a uma gaussiana isotrópica | LeJEPA, LeVJEPA, LeWM, LeWAM |
| frozen encoder | um encoder pré-treinado não é treinado | DINO-WM, RoboJEPA |
| contrastive | [InfoNCE](../api/losses.md) afasta pares que não correspondem | VL-JEPA |
| reward e value | o latent precisa prever reward e value, o que uma constante não consegue | TD-MPC2 |

## De JEPA a world model {#from-jepa-to-world-model}

Quando o context é o presente e o target é o futuro, e o predictor também lê a action tomada, a JEPA se torna um **latent world model**: ela imagina a consequência de uma action no embedding space. O planning então busca as actions cujo resultado imaginado fica mais próximo do embedding de um goal.

<figure class="tw-diagram">
<svg viewBox="0 0 820 330" role="img" aria-labelledby="twd-title twd-desc" xmlns="http://www.w3.org/2000/svg">
<title id="twd-title">Um latent world model planejando em direção a um goal</title>
<desc id="twd-desc">A observation é codificada em um latent z_t. O predictor mapeia z_t e uma action a_t em um próximo latent previsto, e faz o rollout por vários passos. Um cost compara o último latent previsto com a goal image codificada, e o planner ajusta as actions para reduzi-lo.</desc>
<defs>
<marker id="twd-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="twd-head" d="M0 1 L9 5 L0 9 z"/></marker>
<marker id="twd-arrow-accent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="twd-head-accent" d="M0 1 L9 5 L0 9 z"/></marker>
<clipPath id="twd-clip-obs"><rect x="18" y="72" width="76" height="76" rx="12"/></clipPath>
<clipPath id="twd-clip-goal"><rect x="18" y="217" width="76" height="76" rx="12"/></clipPath>
</defs>
<text class="twd-lane" x="18" y="40">imaginar</text>
<text class="twd-lane" x="18" y="206">goal</text>
<image href="../../assets/diagrams/macaw_observation.png" x="18" y="72" width="76" height="76" clip-path="url(#twd-clip-obs)" preserveAspectRatio="xMidYMid slice"/>
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
<text class="twd-caption twd-plan-text" x="548" y="198" text-anchor="middle">planejar: reduzir o cost</text>
<image href="../../assets/diagrams/macaw_goal.png" x="18" y="217" width="76" height="76" clip-path="url(#twd-clip-goal)" preserveAspectRatio="xMidYMid slice"/>
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
<figcaption>Codifique o presente, imagine o futuro uma action de cada vez e planeje buscando as actions cujo resultado imaginado fica mais próximo do goal.</figcaption>
</figure>

## Três formas de escolher actions {#three-ways-to-choose-actions}

Todo world model aqui imagina com o mesmo `rollout`; eles diferem no critério contra o qual o futuro imaginado é avaliado e na origem das actions candidatas.

| planning | avalia futuros imaginados por | propostas vindas de | modelos |
|---|---|---|---|
| goal reaching com [`CEM`](../guides/planning.md) | distância do latent final ao do goal (`cost`) | uma gaussiana, reajustada às melhores | LeWM, DINO-WM, RoboJEPA, qualquer `WorldModel` |
| reward com [`MPPI`](../guides/planning.md#mppi-planning-for-reward) | rewards imaginados mais um value aprendido no horizonte | uma gaussiana e o policy prior | TD-MPC2 |
| policy sampling com `WAMPlanner` | a própria dinâmica do modelo, em direção ao goal | a action head do modelo (`sample_actions`) | DreamZero (closed loop com `DreamZeroPolicy`), LeWAM |

## Classes base {#two-base-classes}

Todo modelo é subclasse de uma de três classes em [`torch_world.models`](../api/models.md).

| classe base | você escreve | você ganha |
|---|---|---|
| `JEPA` | `forward(...) -> {"loss": ...}` | `make_target_encoder`, `update_target(momentum)` |
| `WorldModel` | `encode`, `predict(z, actions)`, `forward` | `rollout`, `cost`, planning com [`CEM`](../guides/planning.md) |
| `WorldActionModel` | um `WorldModel`, mais `sample_actions`, `predict_future` e `imagine` | planning com `WAMPlanner`; `decode` quando os latents têm um decoder |

## Convenções {#conventions}

**Dicts de loss.** `forward` retorna um dict com um `"loss"` escalar para o backpropagation, além de seus componentes nomeados (`"pred_loss"`, `"sigreg_loss"`, ...) para logging.

**O tempo é a dim 1.** Episodes e latents são `(B, T, ...)`. Para `WorldModel`, `predict(z, actions)` mapeia `z[:, t]` e `actions[:, t]` em uma previsão de `z[:, t + 1]`, para todo `t` de uma vez (teacher forcing).

**Actions se alinham com states.** Nos episodes coletados, `actions[:, t]` é aplicada em `states[:, t]` e produz `states[:, t + 1]`. A última action é padding.

**History.** `WorldModel.history` é o número de latents passados que o predictor vê. `rollout` desliza uma window desse tamanho; `None` significa todo o passado.

**Masks são tensors de índices.** Uma mask é um tensor `(B, K)` de índices de tokens a *manter*, e uma estratégia de masking retorna listas delas, uma por block. [`apply_masks`](../api/masks.md) as coleta e empilha múltiplas masks na dim do batch.

**Vídeos são channels-first.** `VisionTransformer` recebe imagens `(B, C, H, W)` ou vídeos `(B, C, T, H, W)`; episodes vindos dos environments são `(B, T, C, H, W)`, então transponha as dims 1 e 2 antes de um video encoder.

## Leitura complementar {#further-reading}

- World Labs, [A functional taxonomy of world models](https://www.worldlabs.ai/blog/taxonomy-of-world-models), 2026: renderers, simulators e planners.
- [World model (artificial intelligence)](https://en.wikipedia.org/wiki/World_model_(artificial_intelligence)), Wikipedia: história e abordagens.
- Ha e Schmidhuber, [World Models](https://arxiv.org/abs/1803.10122), 2018.
- LeCun, [A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf), 2022: a proposta da JEPA.
- Hafner et al., [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551) (PlaNet), 2019, e [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104) (DreamerV3), 2023.
- Schrittwieser et al., [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265) (MuZero), 2020.
