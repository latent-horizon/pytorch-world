# Conceptos

## Qué es un modelo del mundo {#what-a-world-model-is}

Un **modelo del mundo** es un modelo aprendido de cómo cambia un entorno. Un agente actúa, sus acciones cambian un **estado** oculto del mundo y recibe **observaciones** parciales de ese estado. Un modelo del mundo captura este ciclo: a partir de lo observado y de una acción, predice lo que viene después. Un agente capaz de predecir las consecuencias de sus acciones puede ensayarlas antes de ejecutarlas: puede planificar, aprender un comportamiento dentro de su propia imaginación o simular eventos raros en lugar de esperar a que ocurran.

La idea es antigua. Kenneth Craik propuso en 1943 que la mente razona con "modelos a pequeña escala" de la realidad, Schmidhuber entrenó en 1990 redes recurrentes para predecir sus propias entradas futuras, y *World Models* (2018) de Ha y Schmidhuber entrenó agentes dentro de un sueño creado por ellos mismos. Desde entonces el campo se ha dividido en torno a dos preguntas: **qué produce el modelo** y **qué se le entrena a predecir**.

## Qué produce un modelo del mundo {#what-a-world-model-outputs}

Distintos consumidores necesitan distintas proyecciones del mismo ciclo agente–mundo. La [taxonomía funcional](https://www.worldlabs.ai/blog/taxonomy-of-world-models) de World Labs clasifica los modelos del mundo según lo que producen:

| tipo | produce | se evalúa por | ejemplos |
|---|---|---|---|
| **renderer** | observaciones: píxeles para un espectador, a menudo en tiempo real a partir de la entrada del usuario | fidelidad visual | Genie 3, generadores de vídeo interactivos |
| **simulador** | estado: geometría, física y dinámica que un programa o una persona puede consultar | precisión estructural y física | generadores de escenas 3D, gemelos digitales con motor de física |
| **planificador** | acciones: qué hacer a continuación, dada una observación y un objetivo | si el agente alcanza el objetivo | control basado en modelos, modelos visión-lenguaje-acción, world action models |

Los tipos comparten lo que deben saber sobre geometría, física y dinámica, y están convergiendo: los renderers pasan a estar condicionados por acciones, y los planificadores usan renderers de vídeo preentrenados como backbone.

`torch_world` trata de **planificadores**: cada modelo del mundo aquí existe para elegir acciones. Internamente, cada uno sigue renderizando o simulando a su manera, y las familias difieren en lo que predicen.

## Qué predice un modelo del mundo {#what-a-world-model-predicts}

| enfoque | predice | se entrena con | en `torch_world` | en otros lugares |
|---|---|---|---|---|
| **predicción latente** | el *embedding* de la siguiente observación | error de predicción en el espacio de embeddings, más un término contra el colapso | [JEPA](../models/jepa/index.md) (LeWM, RoboJEPA), [DINO-WM](../models/dino/dinowm.md) | V-JEPA 2-AC |
| **latentes equivalentes en valor** | el siguiente latente, la recompensa y el valor de lo que sigue | consistencia latente y aprendizaje por diferencias temporales, sin decoder | [TD-MPC2](../models/tdmpc/tdmpc2.md) | MuZero |
| **dinámica y acciones a la vez** | latentes futuros *y* el bloque de acciones que los produce | predicción latente más flow matching sobre las acciones | [LeWAM](../models/jepa/lewam.md) | world action models sobre difusión de vídeo preentrenada |
| **reconstrucción** | el siguiente estado latente, y los píxeles a partir de él | verosimilitud de las observaciones | | PlaNet, Dreamer |

Predecir píxeles obliga al modelo a dar cuenta de todo lo que hay en el fotograma, incluidas texturas y ruido que no importan para el control; predecir solo embeddings o valores le permite dedicar su capacidad a lo que es predecible y relevante, a costa de necesitar otra forma de evitar que la representación colapse. Los modelos que generan vídeo pagan el precio de los píxeles, pero heredan lo que los grandes conjuntos de datos de vídeo saben sobre cómo se mueve el mundo.

Las familias también incluyen **aprendices de representaciones** que no son modelos del mundo por sí mismos, como I-JEPA, V-JEPA, LeJEPA y [DINOv2](../models/dino/dinov2.md). Preentrenan los encoders sobre los que se construyen los modelos del mundo: DINO-WM planifica sobre features congeladas de DINOv2.

## Predecir representaciones, no píxeles {#predict-representations-not-pixels}

El enfoque latente es la familia más antigua de `torch_world`. Una **joint-embedding predictive architecture** (JEPA) codifica un *contexto* y un *objetivo* y entrena un predictor para transformar el embedding del contexto en el del objetivo. Nunca reconstruye píxeles, así que la capacidad se dedica a lo que es predecible y no a texturas ni ruido.

La dificultad es el colapso: un embedding constante se predice a sí mismo a la perfección. Los modelos de `torch_world` lo evitan de una de estas formas:

| estrategia | cómo | modelos |
|---|---|---|
| profesor EMA | los objetivos provienen de una copia del encoder que se actualiza lentamente, con stop-gradient | I-JEPA, V-JEPA, DINOv2 |
| regularizador de distribución | [SIGReg](../api/losses.md) empuja los embeddings hacia una gaussiana isotrópica | LeJEPA, LeVJEPA, LeWM, LeWAM |
| encoder congelado | un encoder preentrenado que no se entrena en absoluto | DINO-WM, RoboJEPA |
| contrastivo | [InfoNCE](../api/losses.md) separa los pares que no coinciden | VL-JEPA |
| recompensa y valor | el latente debe predecir la recompensa y el valor, algo que una constante no puede hacer | TD-MPC2 |

## De JEPA a modelo del mundo {#from-jepa-to-world-model}

Cuando el contexto es el presente y el objetivo es el futuro, y el predictor también lee la acción tomada, la JEPA se convierte en un **modelo del mundo latente**: imagina la consecuencia de una acción en el espacio de embeddings. La planificación busca entonces las acciones cuyo resultado imaginado queda más cerca del embedding de un objetivo.

<figure class="tw-diagram">
<svg viewBox="0 0 820 330" role="img" aria-labelledby="twd-title twd-desc" xmlns="http://www.w3.org/2000/svg">
<title id="twd-title">Un modelo del mundo latente que planifica hacia un objetivo</title>
<desc id="twd-desc">La observación se codifica en un latente z_t. El predictor transforma z_t y una acción a_t en un latente siguiente predicho, y hace un rollout de varios pasos. Un coste compara el latente final predicho con la imagen objetivo codificada, y el planificador ajusta las acciones para reducirlo.</desc>
<defs>
<marker id="twd-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="twd-head" d="M0 1 L9 5 L0 9 z"/></marker>
<marker id="twd-arrow-accent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="twd-head-accent" d="M0 1 L9 5 L0 9 z"/></marker>
<clipPath id="twd-clip-obs"><rect x="18" y="72" width="76" height="76" rx="12"/></clipPath>
<clipPath id="twd-clip-goal"><rect x="18" y="217" width="76" height="76" rx="12"/></clipPath>
</defs>
<text class="twd-lane" x="18" y="40">imaginar</text>
<text class="twd-lane" x="18" y="206">objetivo</text>
<image href="../../assets/diagrams/macaw_observation.png" x="18" y="72" width="76" height="76" clip-path="url(#twd-clip-obs)" preserveAspectRatio="xMidYMid slice"/>
<rect class="twd-frame" x="18" y="72" width="76" height="76" rx="12"/>
<text class="twd-caption" x="56" y="170" text-anchor="middle">observación</text>
<line class="twd-line" x1="100" y1="110" x2="128" y2="110"/>
<rect class="twd-box" x="130" y="88" width="96" height="44" rx="12"/>
<text x="178" y="116" text-anchor="middle">codificar</text>
<line class="twd-line" x1="226" y1="110" x2="257" y2="110"/>
<circle class="twd-z" cx="288" cy="110" r="29"/>
<text class="twd-math" x="286" y="116" text-anchor="middle">z<tspan class="twd-sub" dy="4">t</tspan></text>
<line class="twd-line" x1="317" y1="110" x2="352" y2="110"/>
<rect class="twd-box twd-box-strong" x="354" y="88" width="104" height="44" rx="12"/>
<text x="406" y="116" text-anchor="middle">predecir</text>
<line class="twd-line" x1="458" y1="110" x2="489" y2="110"/>
<circle class="twd-zhat" cx="520" cy="110" r="29"/>
<text class="twd-math" x="514" y="116" text-anchor="middle">ẑ<tspan class="twd-sub" dy="4">t+1</tspan></text>
<path class="twd-line" d="M520 80 C520 34 406 34 406 86"/>
<text class="twd-caption" x="463" y="40" text-anchor="middle">rollout</text>
<line class="twd-line" x1="549" y1="110" x2="618" y2="110"/>
<text class="twd-caption twd-math" x="583" y="98" text-anchor="middle">ẑ<tspan class="twd-sub" dy="3">T</tspan></text>
<rect class="twd-box twd-box-cost" x="620" y="88" width="120" height="44" rx="12"/>
<text x="680" y="116" text-anchor="middle">coste</text>
<rect class="twd-box" x="370" y="190" width="72" height="34" rx="17"/>
<text class="twd-math" x="402" y="213" text-anchor="middle">a<tspan class="twd-sub" dy="4">t</tspan></text>
<line class="twd-line" x1="406" y1="190" x2="406" y2="134"/>
<path class="twd-line twd-plan" d="M650 134 L650 207 L444 207"/>
<text class="twd-caption twd-plan-text" x="548" y="198" text-anchor="middle">planificar: bajar el coste</text>
<image href="../../assets/diagrams/macaw_goal.png" x="18" y="217" width="76" height="76" clip-path="url(#twd-clip-goal)" preserveAspectRatio="xMidYMid slice"/>
<rect class="twd-frame" x="18" y="217" width="76" height="76" rx="12"/>
<text class="twd-caption" x="56" y="315" text-anchor="middle">imagen objetivo</text>
<line class="twd-line" x1="100" y1="255" x2="128" y2="255"/>
<rect class="twd-box" x="130" y="233" width="96" height="44" rx="12"/>
<text x="178" y="261" text-anchor="middle">codificar</text>
<line class="twd-line" x1="226" y1="255" x2="257" y2="255"/>
<circle class="twd-zgoal" cx="288" cy="255" r="29"/>
<text class="twd-math" x="282" y="261" text-anchor="middle">z<tspan class="twd-sub" dy="4">goal</tspan></text>
<path class="twd-line" d="M317 255 L710 255 L710 134"/>
</svg>
<figcaption>Codifica el presente, imagina el futuro acción a acción y planifica buscando las acciones cuyo resultado imaginado queda más cerca del objetivo.</figcaption>
</figure>

## Tres formas de elegir acciones {#three-ways-to-choose-actions}

Todos los modelos del mundo de aquí imaginan con el mismo `rollout`; se diferencian en contra qué se puntúa el futuro imaginado y de dónde salen las acciones candidatas.

| planificación | puntúa los futuros imaginados por | propuestas de | modelos |
|---|---|---|---|
| alcanzar un objetivo con [`CEM`](../guides/planning.md) | distancia del latente final al del objetivo (`cost`) | una gaussiana, reajustada a las mejores | LeWM, DINO-WM, RoboJEPA, cualquier `WorldModel` |
| recompensa con [`MPPI`](../guides/planning.md#mppi-planning-for-reward) | recompensas imaginadas más un valor aprendido en el horizonte | una gaussiana y el prior de la política | TD-MPC2 |
| muestreo de la política con `LeWAMPlanner` | la propia dinámica del modelo, hacia el objetivo | la cabeza de acciones con flow matching del modelo | LeWAM |

## Dos clases base {#two-base-classes}

Todos los modelos heredan de una de las dos clases de [`torch_world.models`](../api/models.md).

| clase base | escribes | obtienes |
|---|---|---|
| `JEPA` | `forward(...) -> {"loss": ...}` | `make_target_encoder`, `update_target(momentum)` |
| `WorldModel` | `encode`, `predict(z, actions)`, `forward` | `rollout`, `cost`, planificación con [`CEM`](../guides/planning.md) |

## Convenciones {#conventions}

**Dicts de pérdida.** `forward` devuelve un dict con un `"loss"` escalar para hacer backpropagation, más sus componentes con nombre (`"pred_loss"`, `"sigreg_loss"`, ...) para logging.

**El tiempo es la dim 1.** Los episodios y latentes son `(B, T, ...)`. En `WorldModel`, `predict(z, actions)` transforma `z[:, t]` y `actions[:, t]` en una predicción de `z[:, t + 1]`, para todos los `t` a la vez (teacher forcing).

**Las acciones se alinean con los estados.** En los episodios recopilados, `actions[:, t]` se aplica en `states[:, t]` y produce `states[:, t + 1]`. La última acción es relleno.

**Historia.** `WorldModel.history` es el número de latentes pasados que ve el predictor. `rollout` desliza una ventana de esa longitud; `None` significa todo el pasado.

**Las máscaras son tensores de índices.** Una máscara es un tensor `(B, K)` de índices de tokens que se *conservan*, y una estrategia de enmascarado devuelve listas de ellas, una por bloque. [`apply_masks`](../api/masks.md) los recoge y apila varias máscaras en la dim del lote.

**Los vídeos van con los canales primero.** `VisionTransformer` recibe imágenes `(B, C, H, W)` o vídeos `(B, C, T, H, W)`; los episodios de los entornos son `(B, T, C, H, W)`, así que transpón las dims 1 y 2 antes de pasarlos a un encoder de vídeo.

## Lecturas adicionales {#further-reading}

- World Labs, [A functional taxonomy of world models](https://www.worldlabs.ai/blog/taxonomy-of-world-models), 2026: renderers, simuladores y planificadores.
- [World model (artificial intelligence)](https://en.wikipedia.org/wiki/World_model_(artificial_intelligence)), Wikipedia: historia y enfoques.
- Ha y Schmidhuber, [World Models](https://arxiv.org/abs/1803.10122), 2018.
- LeCun, [A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf), 2022: la propuesta de JEPA.
- Hafner et al., [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551) (PlaNet), 2019, y [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104) (DreamerV3), 2023.
- Schrittwieser et al., [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265) (MuZero), 2020.
