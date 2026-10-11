---
title: pytorch-world
hide:
  - navigation
  - toc
---

# pytorch-world

<div class="tw-hero" markdown>

![pytorch-world](assets/banner-dark.png)

World models for PyTorch. Learn how a world evolves, then plan by imagining what comes next. Shared parts keep every model short and easy to compare.

<div class="tw-install" markdown>

```bash
uv add torch_world
```

</div>

<div class="tw-actions" markdown>
[Get started](getting-started/quickstart.md){ .md-button .md-button--primary }
[Models](models/index.md){ .md-button }
[API](api/index.md){ .md-button }
</div>

</div>

A **world model** learns how an environment evolves: from an observation and an action, it predicts what happens next. With one, an agent can plan by imagining the outcomes of its actions before taking them.

`torch_world` does for world models what PyTorch Geometric did for graph neural networks. Models are organized into **families** that share a way of learning, and every model is built from the same layers, objectives, planners, environments and datasets. The first family is **JEPA**, which predicts the future in a learned representation space instead of in pixels. More families and models are on the way.

<div class="grid cards" markdown>

-   :material-family-tree:{ .lg } **Model families**

    ---

    Models grouped by how they learn. JEPA comes first, with seven models from image pretraining to multi-robot control. More families are coming.

    [:octicons-arrow-right-24: Models](models/index.md)

-   :material-earth:{ .lg } **One world-model interface**

    ---

    `encode`, `predict` and `rollout` are the same for every world model, so they can be trained, compared and planned with interchangeably.

    [:octicons-arrow-right-24: Concepts](getting-started/concepts.md)

-   :material-map-marker-path:{ .lg } **Planning**

    ---

    Search for actions inside a model's imagination. The cross-entropy method works with any world model, with no extra code.

    [:octicons-arrow-right-24: Planning guide](guides/planning.md)

-   :material-robot-industrial:{ .lg } **Environments**

    ---

    Batched pixel worlds that render on the GPU, and MuJoCo pick-and-place with UR10e and SO-101 arms.

    [:octicons-arrow-right-24: Environments](guides/environments.md)

-   :material-database-outline:{ .lg } **Datasets**

    ---

    Windows over collected experience for training dynamics, plus medical images and questions for vision-language models.

    [:octicons-arrow-right-24: Datasets](datasets/index.md)

-   :material-cube-outline:{ .lg } **Building blocks**

    ---

    Vision transformers for images and video, conditioned predictors, masking strategies and objectives, shared by every family.

    [:octicons-arrow-right-24: API reference](api/index.md)

</div>

## Thirty seconds

```python
from torch_world.envs import PointMass
from torch_world.models import LeWM
from torch_world.nn import CLSEncoder
from torch_world.planning import CEM

model = LeWM(CLSEncoder(img_size=32, patch_size=8, embed_dim=64), action_dim=2, embed_dim=64)
episodes = PointMass(batch_size=64).collect(4)                 # pixels (64, 4, 3, 32, 32), actions (64, 4, 2)
loss = model(episodes["pixels"], episodes["actions"])["loss"]  # prediction + SIGReg

model.eval()
z, goal = model.encode(episodes["pixels"][:, :1]), model.encode(episodes["pixels"][:, -1:])[:, 0]
actions = CEM(horizon=5, action_dim=2)(model, z, goal)         # (64, 5, 2)
```

This trains LeWM, a JEPA world model, on a point mass and plans toward a goal image. Read the [quick start](getting-started/quickstart.md) for what each line does, or open the [tour notebook](examples/intro.ipynb).
