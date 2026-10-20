# Datasets

`torch_world.datasets` holds the data the models train on: windows over collected episodes for world models, and medical images and questions for vision-language models. Every dataset is a standard `torch.utils.data.Dataset` that returns dicts of tensors, so it plugs into a `DataLoader` as is.

<div class="grid cards" markdown>

-   :material-chart-timeline-variant:{ .lg } **[Trajectories](trajectories.md)**

    ---

    Sliding windows over episodes collected in any environment, for training world models.

    `TrajectoryDataset`

-   :material-microscope:{ .lg } **[MedMNIST](medmnist.md)**

    ---

    Five MedMNIST v2 image classification datasets, downloaded and checked on first use.

    `MedMNIST`

-   :material-chat-question-outline:{ .lg } **[Medical VQA](medical-vqa.md)**

    ---

    Templated visual question answering over MedMNIST, in the style of the MMBU challenge.

    `MedicalVQA`

</div>

## At a glance

| dataset | each item | source | download | used by |
|---|---|---|---|---|
| [`TrajectoryDataset`](trajectories.md) | a window of pixels, actions and states | any [environment](../guides/environments.md), or your own episodes | none | [LeWM](../models/jepa/lewm.md), [DINO-WM](../models/dino/dinowm.md) |
| [`MedMNIST`](medmnist.md) | an image and its label | [MedMNIST v2](https://arxiv.org/abs/2110.14795), CC BY 4.0 | Zenodo, MD5-checked | image classification, linear probes |
| [`MedicalVQA`](medical-vqa.md) | an image, its caption, a question and its answer | MedMNIST v2 + question templates | Zenodo, MD5-checked | [VL-JEPA](../models/jepa/vljepa.md) |

Downloads are cached in `~/.cache/torch_world/` unless you pass another `root`.
