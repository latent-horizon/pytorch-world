# Medical VQA

`MedicalVQA` turns one or more [MedMNIST](medmnist.md) datasets into visual question answering, with templated questions in the style of the [MMBU challenge](https://akiranishii.github.io/mmbu-challenge/). It is the data of the [VL-JEPA notebook](../examples/vqa/vljepa.ipynb).

## Questions

Every image yields up to four questions, depending on its source dataset:

| kind | question | answer | when |
|---|---|---|---|
| modality | *what imaging modality is this ?* | *dermatoscopy* | always |
| finding | *what finding is shown ?* | *melanoma* | more than two classes |
| verification | *is this a melanoma ?* | *yes* / *no*, true half of the time | more than two classes |
| malignancy | *is the lesion malignant ?* | *yes* / *no* | datasets with malignant classes |

Every image also has a caption, such as *dermatoscopy showing a melanoma*, for caption pretraining. The false claims of verification questions are drawn with `seed`, so the dataset is the same on every run.

## Usage

```python
from torch.utils.data import DataLoader, WeightedRandomSampler

from torch_world.datasets import MedicalVQA

vqa = MedicalVQA(["bloodmnist", "dermamnist", "breastmnist"], split="train", size=64)
vqa[0]   # {"image", "caption", "question", "answer", "kind"}

sampler = WeightedRandomSampler(vqa.weights(), num_samples=len(vqa))
loader = DataLoader(vqa, batch_size=256, sampler=sampler)
```

| member | |
|---|---|
| `images` | every image once, uint8 `(N_images, 3, size, size)` |
| `captions` | one caption per image |
| `answers` | every answer the templates can produce, sorted: the candidates for retrieval |
| `texts` | every caption, question and answer the templates can produce: a vocabulary for a tokenizer |
| `weights()` | per-item sampling weights that give each source dataset the same total weight |
| `info` | a table of the source datasets and of the questions of each kind |
| `examples(per_source)` | a few images of each source and their findings, for `plots.images` |
| `pixels(items)` | the uint8 images of the given items, even after `encode` |
| `KINDS` | `("modality", "finding", "verification", "malignancy")` |

## Cached encoder features

A frozen image encoder only needs to run once per image, not once per question. `encode` caches its outputs, and items then carry the features in place of the images:

```python
from torch_world.nn import PatchTokens

encoder = PatchTokens.from_hub("dinov2_vits14", size=112).cuda()
vqa = vqa.encode(encoder)            # float16 features, cached
vqa[0]["image"].shape                # (64, 384): 8 x 8 patch tokens of DINOv2 ViT-S/14
```

## Answering by retrieval

Models that predict answer embeddings, like [VL-JEPA](../models/jepa/vljepa.md), answer by finding the nearest of `vqa.answers`. [`evaluation.vqa`](../api/evaluation.md) does this for every question and scores the answers per kind:

```python
from torch_world.evaluation import vqa as evaluate
from torch_world.nn import WordTokenizer

tokenizer = WordTokenizer(vqa.texts)
results = evaluate(model, test_set, tokenizer)
results.accuracy()     # {"modality": ..., "finding": ..., ..., "all": ...}
results.majority()     # the always-the-most-common-answer baseline per kind
plots.bars(results.scores("VL-JEPA"), baseline="majority answer")
plots.answers(*results.cards())
```

## Reference

[`MedicalVQA`](../api/datasets.md#torch_world.datasets.MedicalVQA) in the API reference.
