# Modelos

Os modelos são agrupados em **famílias**. Modelos de uma mesma família compartilham uma forma de aprender e uma classe base, então podem ser trocados e comparados nos mesmos dados, environments e planners. As famílias são **JEPA**, **DINO**, **TD-MPC** e **world action models**; mais famílias e modelos serão adicionados.

<div class="grid cards tw-models tw-models--families">

-   :material-vector-link:{ .lg } **[JEPA](jepa/index.md)**

    Prevê a *representação* de um target, nunca seus pixels; com actions, um latent world model.

-   :material-school-outline:{ .lg } **[DINO](dino/index.md)**

    Self-distillation sobre prototypes aprendidos, e world models sobre suas frozen features.

-   :material-chart-timeline-variant:{ .lg } **[TD-MPC](tdmpc/index.md)**

    Latent dynamics sem decoder, treinada por TD learning e planejada para maximizar o return.

-   :material-movie-open-play-outline:{ .lg } **[World action models](wam/index.md)**

    Fazem o denoise do vídeo futuro e das actions que o produzem em conjunto; a policy é o próprio modelo.

</div>

## Todos os modelos em um relance {#every-model-at-a-glance}

| família | grupo | modelo | input | classe base | notebook |
|---|---|---|---|---|---|
| JEPA | Action | [LeWM](jepa/lewm.md) | frames + actions | `WorldModel` | [action/lewm](../examples/action/lewm.ipynb) |
| JEPA | Action | [LeWAM](jepa/lewam.md) | frames + actions (+ goal image) | `WorldActionModel` | [action/lewam](../examples/action/lewam.ipynb) |
| JEPA | Action | [RoboJEPA](jepa/robojepa.md) | frames multi-view + actions + states | `WorldModel` | [action/robojepa](../examples/action/robojepa.ipynb) |
| JEPA | Vision | [I-JEPA](jepa/ijepa.md) | imagens | `JEPA` | [vision/ijepa](../examples/vision/ijepa.ipynb) |
| JEPA | Vision | [LeJEPA](jepa/lejepa.md) | imagens | `JEPA` | [vision/lejepa](../examples/vision/lejepa.ipynb) |
| JEPA | Vision | [V-JEPA](jepa/vjepa.md) | vídeo | `JEPA` | [vision/vjepa](../examples/vision/vjepa.ipynb) |
| JEPA | Vision | [LeVJEPA](jepa/levjepa.md) | vídeo | `JEPA` | [vision/levjepa](../examples/vision/levjepa.ipynb) |
| JEPA | Vision-language | [VL-JEPA](jepa/vljepa.md) | imagens ou vídeo + texto | `JEPA` | [vqa/vljepa](../examples/vqa/vljepa.ipynb) |
| DINO | Action | [DINO-WM](dino/dinowm.md) | frames + actions + proprioception | `WorldModel` | [action/dinowm](../examples/action/dinowm.ipynb) |
| DINO | Vision | [DINOv2](dino/dinov2.md) | imagens | `nn.Module` | [vision/dinov2](../examples/vision/dinov2.ipynb) |
| TD-MPC | Control | [TD-MPC2](tdmpc/tdmpc2.md) | states ou frames + actions + rewards | `WorldModel` | [action/tdmpc2](../examples/action/tdmpc2.ipynb) |
| WAM | Action | [DreamZero](wam/dreamzero.md) | latents de video-VAE + actions (+ states, goal image) | `WorldActionModel` | [action/dreamzero](../examples/action/dreamzero.ipynb) |

O `forward` de todo modelo retorna um dict com um `"loss"` escalar. Todo `WorldModel` também tem `encode`, `predict` e `rollout`, e planeja com [`CEM`](../guides/planning.md) sem configuração extra; o TD-MPC2 planeja para reward com `MPPI`. Um `WorldActionModel` também amostra suas próprias actions (`sample_actions`) e planeja com `WAMPlanner`. Para adicionar um modelo, veja [Traga seu próprio modelo](../guides/custom-model.md).
