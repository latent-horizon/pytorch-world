import random
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


@dataclass
class VQAResults:
    """Predicted and true answers of every question in ``dataset``, with the kind of each question."""

    dataset: Dataset
    predictions: list[str]
    truths: list[str]
    kinds: list[str]

    def accuracy(self) -> dict[str, float]:
        """Accuracy per question kind, and over all questions."""
        out = {}
        for kind in [*dict.fromkeys(self.kinds), "all"]:
            pairs = [(p, t) for p, t, k in zip(self.predictions, self.truths, self.kinds) if kind in (k, "all")]
            out[kind] = sum(p == t for p, t in pairs) / len(pairs)
        return out

    def majority(self) -> dict[str, float]:
        """Accuracy of always giving the most common true answer of each kind."""
        out = {}
        for kind in dict.fromkeys(self.kinds):
            answers = [t for t, k in zip(self.truths, self.kinds) if k == kind]
            out[kind] = max(map(answers.count, set(answers))) / len(answers)
        return out

    def scores(self, name: str = "model") -> dict[str, dict[str, float]]:
        """Accuracy beside the majority baseline per kind, as ``plots.bars`` expects."""
        majority = self.majority()
        return {
            kind: {name: acc, "majority answer": majority[kind]}
            for kind, acc in self.accuracy().items()
            if kind in majority
        }

    def cards(self, per_kind: int = 2, seed: int = 0) -> tuple[Tensor, list[str], list[str], list[str]]:
        """``per_kind`` questions of each kind, a mistake first when there is one, as ``plots.answers`` expects."""
        order = list(range(len(self.truths)))
        random.Random(seed).shuffle(order)
        picks = []
        for kind in dict.fromkeys(self.kinds):
            right = [i for i in order if self.kinds[i] == kind and self.predictions[i] == self.truths[i]]
            wrong = [i for i in order if self.kinds[i] == kind and self.predictions[i] != self.truths[i]]
            picks += (wrong[:1] + right)[:per_kind]
        questions = [self.dataset[i]["question"].replace(" ?", "?") for i in picks]
        return (
            self.dataset.pixels(picks),
            questions,
            [self.predictions[i] for i in picks],
            [self.truths[i] for i in picks],
        )

    def __repr__(self) -> str:
        majority = self.majority()
        rows = [f"{'kind':14s}{'accuracy':>10s}{'majority':>10s}{'questions':>11s}"]
        for kind, acc in self.accuracy().items():
            n = len(self.kinds) if kind == "all" else self.kinds.count(kind)
            base = f"{majority[kind]:10.3f}" if kind in majority else " " * 10
            rows.append(f"{kind:14s}{acc:10.3f}{base}{n:11d}")
        return "\n".join(rows)


def compare(results: dict[str, VQAResults], baseline: str = "majority answer") -> dict[str, dict[str, float]]:
    """Accuracy per kind of several evaluations, e.g. on train and test, beside the majority baseline of the last one,
    as ``plots.bars`` expects."""
    majority = list(results.values())[-1].majority()
    return {
        kind: {name: r.accuracy()[kind] for name, r in results.items()} | {baseline: base}
        for kind, base in majority.items()
    }


@torch.no_grad()
def vqa(
    model,
    dataset: Dataset,
    tokenizer,
    candidates: list[str] | None = None,
    batch_size: int = 1024,
    shuffle_images: bool = False,
    generator: torch.Generator | None = None,
) -> VQAResults:
    """Answer every question of ``dataset`` by retrieval over ``candidates`` (by default ``dataset.answers``).

    ``model`` follows the ``VLJEPA`` interface (``embed_text`` and ``retrieve``) and items carry ``image``,
    ``question``, ``answer`` and ``kind``. With ``shuffle_images``, each question is asked about a
    random image of the whole dataset: a control that measures how much the answers come from the image rather than
    from the question.
    """
    candidates = candidates or dataset.answers
    bank = model.embed_text(*tokenizer(candidates))
    predictions, truths, kinds = [], [], []
    for batch in DataLoader(_Shuffled(dataset, generator) if shuffle_images else dataset, batch_size=batch_size):
        x = batch["image"].to(bank.device)
        x = x.float() if x.is_floating_point() else x
        predictions += [candidates[i] for i in model.retrieve(x, bank, *tokenizer(batch["question"])).tolist()]
        truths += batch["answer"]
        kinds += batch["kind"]
    return VQAResults(dataset, predictions, truths, kinds)


class _Shuffled(Dataset):
    def __init__(self, dataset: Dataset, generator: torch.Generator | None = None):
        self.dataset, self.order = dataset, torch.randperm(len(dataset), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i: int) -> dict:
        return {**self.dataset[i], "image": self.dataset[self.order[i]]["image"]}
