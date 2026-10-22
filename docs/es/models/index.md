# Modelos

Los modelos se agrupan en **familias**. Los modelos de una familia comparten una forma de aprender y una clase base, de modo que pueden intercambiarse y compararse con los mismos datos, entornos y planificadores. Las familias son **JEPA**, **DINO**, **TD-MPC** y **world action models**; se añadirán más familias y modelos.

<div class="grid cards tw-models tw-models--families">

-   :material-vector-link:{ .lg } **[JEPA](jepa/index.md)**

    Predicen la *representación* de un objetivo, nunca sus píxeles; con acciones, un world model latente.

-   :material-school-outline:{ .lg } **[DINO](dino/index.md)**

    Autodestilación sobre prototipos aprendidos, y world models sobre sus features congeladas.

-   :material-chart-timeline-variant:{ .lg } **[TD-MPC](tdmpc/index.md)**

    Dinámica latente sin decoder entrenada con aprendizaje TD y planificada para maximizar el retorno.

-   :material-movie-open-play-outline:{ .lg } **[World action models](wam/index.md)**

    Eliminan el ruido del vídeo futuro y de las acciones que lo producen a la vez; la política es el modelo.

</div>

## Todos los modelos de un vistazo {#every-model-at-a-glance}

| familia | grupo | modelo | entrada | clase base | notebook |
|---|---|---|---|---|---|
| JEPA | Acción | [LeWM](jepa/lewm.md) | frames + acciones | `WorldModel` | [action/lewm](../examples/action/lewm.ipynb) |
| JEPA | Acción | [LeWAM](jepa/lewam.md) | frames + acciones (+ imagen objetivo) | `WorldActionModel` | [action/lewam](../examples/action/lewam.ipynb) |
| JEPA | Acción | [RoboJEPA](jepa/robojepa.md) | frames multivista + acciones + estados | `WorldModel` | [action/robojepa](../examples/action/robojepa.ipynb) |
| JEPA | Visión | [I-JEPA](jepa/ijepa.md) | imágenes | `JEPA` | [vision/ijepa](../examples/vision/ijepa.ipynb) |
| JEPA | Visión | [LeJEPA](jepa/lejepa.md) | imágenes | `JEPA` | [vision/lejepa](../examples/vision/lejepa.ipynb) |
| JEPA | Visión | [V-JEPA](jepa/vjepa.md) | vídeo | `JEPA` | [vision/vjepa](../examples/vision/vjepa.ipynb) |
| JEPA | Visión | [LeVJEPA](jepa/levjepa.md) | vídeo | `JEPA` | [vision/levjepa](../examples/vision/levjepa.ipynb) |
| JEPA | Visión-lenguaje | [VL-JEPA](jepa/vljepa.md) | imágenes o vídeo + texto | `JEPA` | [vqa/vljepa](../examples/vqa/vljepa.ipynb) |
| DINO | Acción | [DINO-WM](dino/dinowm.md) | frames + acciones + propiocepción | `WorldModel` | [action/dinowm](../examples/action/dinowm.ipynb) |
| DINO | Visión | [DINOv2](dino/dinov2.md) | imágenes | `nn.Module` | [vision/dinov2](../examples/vision/dinov2.ipynb) |
| TD-MPC | Control | [TD-MPC2](tdmpc/tdmpc2.md) | estados o frames + acciones + recompensas | `WorldModel` | [action/tdmpc2](../examples/action/tdmpc2.ipynb) |
| WAM | Acción | [DreamZero](wam/dreamzero.md) | latentes de video-VAE + acciones (+ estados, imagen objetivo) | `WorldActionModel` | [action/dreamzero](../examples/action/dreamzero.ipynb) |

El `forward` de cada modelo devuelve un dict con un `"loss"` escalar. Todo `WorldModel` tiene además `encode`, `predict` y `rollout`, y planifica con [`CEM`](../guides/planning.md) sin configuración adicional; TD-MPC2 planifica para maximizar la recompensa con `MPPI`. Un `WorldActionModel` también muestrea sus propias acciones (`sample_actions`) y planifica con `WAMPlanner`. Para añadir un modelo, consulta [Trae tu propio modelo](../guides/custom-model.md).
