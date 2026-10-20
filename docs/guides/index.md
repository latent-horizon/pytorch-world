# Guides

Every part of `torch_world` is replaceable: models, environments, data and planners meet at small, documented interfaces. The guides fall into three groups.

## Bring your own

Plug your own component into the library. Each guide states the interface, gives a complete example, and shows how to run it with the rest of the library.

<div class="grid cards">

-   :material-cube-outline:{ .lg } **[Model](custom-model.md)**

    ---

    Subclass `WorldModel` or `JEPA`: implement `encode`, `predict` and `forward`, and get rollouts, planning and benchmarks.

-   :material-earth:{ .lg } **[Environment](custom-environment.md)**

    ---

    Register a variant of a built-in task, or subclass `BenchmarkEnv` for new dynamics.

-   :material-database-outline:{ .lg } **[Dataset](custom-dataset.md)**

    ---

    Train on your own episodes: a dict of tensors `(N, T, ...)`, or any `Dataset` of dicts.

-   :material-map-marker-path:{ .lg } **[Planner](custom-planner.md)**

    ---

    Any callable `planner(model, z, goal)` that returns an action sequence.

</div>

## Train and evaluate

| guide | covers |
|---|---|
| [Training](training.md) | the training loop, EMA targets, BatchNorm statistics, frozen encoders |
| [Masking](masking.md) | image blocks, video tubes, frame-causal attention |
| [Planning](planning.md) | rollouts, costs, CEM, closed-loop control, goal-reaching evaluation |
| [Metrics](metrics.md) | probes, collapse checks, prediction error, control metrics |
| [Figures](plots.md) | the plotting functions used in the notebooks |

## Benchmarks

| guide | covers |
|---|---|
| [Benchmark experiments](benchmark-environments.md) | backends, rendering quality, `run_experiment`, saved runs, GT references |
| [Pixel worlds](environments.md) | the original `PointMass`, `Macaw` and `RobotCube` interfaces |

The [Environments](../environments/index.md) page lists every registered task by category.
