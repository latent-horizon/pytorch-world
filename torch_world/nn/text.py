import torch
from torch import Tensor, nn

from torch_world.nn.transformer import Transformer


class WordTokenizer:
    """Whitespace tokenizer over a fixed vocabulary, for small closed-vocabulary text such as templated QA.

    Id 0 is padding. ``tokenizer(texts)`` returns ``(ids, mask)`` of shape ``(B, L)``, padded to the longest text.
    With a pretrained text model, use its own tokenizer instead.
    """

    def __init__(self, texts: list[str], device=None):
        self.vocabulary = ["<pad>"] + sorted({w for t in texts for w in t.split()})
        self.index = {w: i for i, w in enumerate(self.vocabulary)}
        self.device = device

    def __len__(self) -> int:
        return len(self.vocabulary)

    def __call__(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        ids = [[self.index[w] for w in t.split()] for t in texts]
        n = max(map(len, ids))
        ids = torch.tensor([t + [0] * (n - len(t)) for t in ids], device=self.device)
        return ids, ids != 0

    def to(self, device) -> "WordTokenizer":
        self.device = device
        return self


class TextEncoder(nn.Module):
    """Token embedding, a bidirectional transformer and mean pooling over non-padding tokens: ``(B, dim)``.

    A small stand-in for a pretrained text encoder such as the EmbeddingGemma Y-encoder of VL-JEPA.
    """

    def __init__(self, vocab_size: int, dim: int = 128, depth: int = 2, num_heads: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.transformer = Transformer(dim, depth, num_heads)

    def forward(self, ids: Tensor, mask: Tensor) -> Tensor:
        h = self.transformer(self.embed(ids), attn_mask=mask[:, None, None, :])
        return (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
