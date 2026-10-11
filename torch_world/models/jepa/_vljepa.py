import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_world.losses.contrastive import info_nce
from torch_world.models.base import JEPA


class VLJEPA(JEPA):
    """VL-JEPA: predict the embedding of a target text from visual tokens and an optional text query.

    A frozen X-encoder turns images or videos into visual tokens. The predictor attends bidirectionally over the
    projected visual tokens followed by the query tokens, and the non-padding outputs are average-pooled into a
    predicted embedding. The Y-encoder embeds the target text. Linear heads map both into a shared space, and the
    predictor and Y-encoder are trained jointly with bidirectional InfoNCE. The Y-encoder trains at
    ``y_lr_mult`` times the base learning rate (see ``param_groups``). Text is produced only on demand by a
    separate decoder; ``segment`` picks the decoding points of an embedding stream (selective decoding).

    The paper does not report the InfoNCE temperature, so it is learnable here, initialized at 0.07 as in CLIP.

    Chen et al., "VL-JEPA: Joint Embedding Predictive Architecture for Vision-language", ICLR 2026.
    Paper: https://arxiv.org/abs/2512.10942
    Code: not released; this follows the paper.

    Args:
        x_encoder: frozen module mapping visual input to tokens ``(B, N, vision_dim)`` (V-JEPA 2 ViT-L in the paper).
        predictor: module mapping ``(tokens, attn_mask=...)`` to tokens of the same shape (last 8 Llama-3.2-1B layers).
        query_embed: token embedding of the predictor's language model.
        y_encoder: module mapping ``(ids, mask)`` to pooled text embeddings ``(B, text_dim)`` (EmbeddingGemma-300M).
    """

    def __init__(
        self,
        x_encoder: nn.Module,
        predictor: nn.Module,
        query_embed: nn.Embedding,
        y_encoder: nn.Module,
        vision_dim: int,
        predictor_dim: int,
        text_dim: int,
        embed_dim: int = 1536,
        temperature: float = 0.07,
        y_lr_mult: float = 0.05,
    ):
        super().__init__()
        self.x_encoder = x_encoder.requires_grad_(False).eval()
        self.vision_proj = nn.Linear(vision_dim, predictor_dim)
        self.query_embed = query_embed
        self.predictor = predictor
        self.pred_head = nn.Linear(predictor_dim, embed_dim)
        self.y_encoder = y_encoder
        self.y_head = nn.Linear(text_dim, embed_dim)
        self.log_scale = nn.Parameter(torch.tensor(math.log(1 / temperature)))
        self.y_lr_mult = y_lr_mult

    def train(self, mode: bool = True):
        super().train(mode)
        self.x_encoder.eval()
        return self

    def predict(self, x: Tensor, query_ids: Tensor | None = None, query_mask: Tensor | None = None) -> Tensor:
        """Predicted target embedding ``(B, embed_dim)``; ``query_mask`` is ``True`` on non-padding query tokens."""
        with torch.no_grad():
            v = self.x_encoder(x)
        tokens = self.vision_proj(v)
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        if query_ids is not None:
            tokens = torch.cat([tokens, self.query_embed(query_ids)], dim=1)
            mask = torch.cat(
                [mask, query_mask if query_mask is not None else torch.ones_like(query_ids, dtype=torch.bool)], dim=1
            )
        h = self.predictor(tokens, attn_mask=mask[:, None, None, :])
        pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        return self.pred_head(pooled)

    def embed_text(self, ids: Tensor, mask: Tensor | None = None) -> Tensor:
        return self.y_head(self.y_encoder(ids, mask))

    @torch.no_grad()
    def retrieve(
        self, x: Tensor, bank: Tensor, query_ids: Tensor | None = None, query_mask: Tensor | None = None
    ) -> Tensor:
        """Index ``(B,)`` of the closest row of ``bank`` (candidate text embeddings from ``embed_text``) by cosine."""
        pred = F.normalize(self.predict(x, query_ids, query_mask), dim=-1)
        return (pred @ F.normalize(bank, dim=-1).T).argmax(1)

    def forward(
        self,
        x: Tensor,
        target_ids: Tensor,
        target_mask: Tensor | None = None,
        query_ids: Tensor | None = None,
        query_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        pred = self.predict(x, query_ids, query_mask)
        target = self.embed_text(target_ids, target_mask)
        return {"loss": info_nce(pred, target, 1 / self.log_scale.exp())}

    def param_groups(self, lr: float, **kwargs) -> list[dict]:
        """Optimizer groups with the Y-encoder at ``y_lr_mult * lr``; the frozen X-encoder is excluded."""
        y = {id(p) for p in self.y_encoder.parameters()}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in y]
        return [
            {"params": rest, "lr": lr, **kwargs},
            {"params": list(self.y_encoder.parameters()), "lr": lr * self.y_lr_mult, **kwargs},
        ]

    @staticmethod
    def segment(stream: Tensor, num_segments: int) -> Tensor:
        """Selective decoding points: midpoints of ``num_segments`` contiguous segments of a ``(T, D)`` stream.

        Segments come from agglomerative clustering with temporal connectivity and Ward linkage.
        """
        starts, sizes, means = list(range(len(stream))), [1] * len(stream), list(stream.float())
        while len(starts) > num_segments:
            costs = [
                sizes[i] * sizes[i + 1] / (sizes[i] + sizes[i + 1]) * (means[i] - means[i + 1]).square().sum()
                for i in range(len(starts) - 1)
            ]
            i = int(torch.stack(costs).argmin())
            n = sizes[i] + sizes[i + 1]
            means[i] = (sizes[i] * means[i] + sizes[i + 1] * means[i + 1]) / n
            sizes[i] = n
            del starts[i + 1], sizes[i + 1], means[i + 1]
        return torch.tensor([s + (n - 1) // 2 for s, n in zip(starts, sizes)])
