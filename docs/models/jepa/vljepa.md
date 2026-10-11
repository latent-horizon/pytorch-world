# VL-JEPA

> Chen et al., *VL-JEPA: Joint Embedding Predictive Architecture for Vision-language*, ICLR 2026. [Paper](https://arxiv.org/abs/2512.10942) · Code not released · [Notebook](../../examples/vqa/vljepa.ipynb)

VL-JEPA predicts the *embedding* of a target text from visual tokens and an optional text query, so it never generates tokens during training.

- A frozen **X-encoder** turns images or videos into visual tokens.
- The **predictor** attends bidirectionally over the projected visual tokens followed by the query tokens; the non-padding outputs are average-pooled into a predicted embedding.
- The **Y-encoder** embeds the target text.
- Linear heads map both into a shared space, trained with bidirectional InfoNCE. The temperature is learnable, initialized at 0.07.

Text comes out only on demand: by a separate decoder, or by retrieval over candidate answers. `VLJEPA.segment` picks the decoding points of an embedding stream (selective decoding).

## Usage

```python
from torch import nn

from torch_world.models import VLJEPA
from torch_world.nn import TextEncoder, Transformer, WordTokenizer

tokenizer = WordTokenizer(vqa.texts)                   # closed vocabulary of templated QA
model = VLJEPA(
    nn.Identity(),                                     # X-encoder outputs cached with MedicalVQA.encode
    Transformer(256, 4, 8), nn.Embedding(len(tokenizer), 256), TextEncoder(len(tokenizer)),
    vision_dim=384, predictor_dim=256, text_dim=128, embed_dim=128, y_lr_mult=0.1,
)
opt = torch.optim.AdamW(model.param_groups(3e-4, weight_decay=0.01))

loss = model(images, *tokenizer(answers), *tokenizer(questions))["loss"]

# answer by retrieval over candidate answers
bank = model.embed_text(*tokenizer(vqa.answers))
answer = model.retrieve(images, bank, *tokenizer(questions))   # (B,) indices into vqa.answers
```

[`TextEncoder`](../../api/nn.md#torch_world.nn.TextEncoder) and [`WordTokenizer`](../../api/nn.md#torch_world.nn.WordTokenizer) are small stand-ins for a pretrained text model and its tokenizer. [`evaluation.vqa`](../../datasets/medical-vqa.md#answering-by-retrieval) answers and scores a whole dataset.

| argument | in the paper |
|---|---|
| `x_encoder` | V-JEPA 2 ViT-L, frozen |
| `predictor`, `query_embed` | the last 8 layers and the token embedding of Llama-3.2-1B |
| `y_encoder` | EmbeddingGemma-300M |

The [notebook](../../examples/vqa/vljepa.ipynb) trains it on medical VQA over MedMNIST with a frozen DINOv2 X-encoder; see [`MedicalVQA`](../../datasets/medical-vqa.md).

## Reference

Arguments and methods: [`VLJEPA`](../../api/models.md#torch_world.models.VLJEPA) in the API reference.
