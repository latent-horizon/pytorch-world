from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class Probe:
    """Test ``score`` of a probe (accuracy for classes, R² for regression) and its test ``predictions``."""

    score: float
    predictions: Tensor


def r2_score(pred: Tensor, target: Tensor) -> float:
    """Coefficient of determination of ``(N, K)`` predictions, averaged over the ``K`` targets."""
    pred, target = pred.reshape(len(pred), -1), target.reshape(len(target), -1).to(pred)
    residual = (target - pred).square().sum(0)
    total = (target - target.mean(0)).square().sum(0).clamp_min(1e-12)
    return (1 - residual / total).mean().item()


def linear_probe(
    train_x: Tensor, train_y: Tensor, test_x: Tensor, test_y: Tensor, ridge: float = 1e-3, iterations: int = 100
) -> Probe:
    """Fit a linear map from frozen features ``(N, D)`` to targets and score it on the test split.

    Integer targets ``(N,)`` are classes: multinomial logistic regression fit with full-batch L-BFGS, scored by
    accuracy. Float targets ``(N,)`` or ``(N, K)`` are regressed in closed form, scored by R². Features are
    standardized with the train statistics, and ``ridge`` is an L2 penalty on the weights. This is the linear
    evaluation of I-JEPA and LeJEPA, and the physical probing of LeWM.
    """
    mean, std = train_x.mean(0), train_x.std(0).clamp_min(1e-6)
    x = F.pad((train_x.float() - mean) / std, (0, 1), value=1.0)
    t = F.pad((test_x.float() - mean) / std, (0, 1), value=1.0)
    if train_y.is_floating_point():
        y = train_y.reshape(len(x), -1).to(x)
        eye = torch.eye(x.size(1), dtype=x.dtype, device=x.device)
        with torch.no_grad():
            pred = (t @ torch.linalg.solve(x.T @ x + ridge * len(x) * eye, x.T @ y)).view_as(test_y)
        return Probe(r2_score(pred, test_y), pred)
    num_classes = int(max(train_y.max(), test_y.max())) + 1
    w = torch.zeros(x.size(1), num_classes, device=x.device, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=iterations, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x @ w, train_y) + ridge * w[:-1].square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        pred = (t @ w).argmax(1)
    return Probe((pred == test_y).float().mean().item(), pred)


@torch.no_grad()
def knn_accuracy(
    train_x: Tensor,
    train_y: Tensor,
    test_x: Tensor,
    test_y: Tensor,
    k: int = 20,
    temperature: float = 0.07,
    center: bool = False,
) -> Probe:
    """Weighted k-nearest-neighbor classification on cosine similarity: each of the ``k`` nearest train features votes
    for its class with weight ``exp(similarity / temperature)``. Needs no training, as reported by DINO and
    LeJEPA. ``center`` subtracts the mean train feature first, so a component shared by every input does not
    dominate the cosine similarity.

    Wu et al., "Unsupervised Feature Learning via Non-Parametric Instance Discrimination", CVPR 2018.
    Paper: https://arxiv.org/abs/1805.01978
    """
    num_classes = int(max(train_y.max(), test_y.max())) + 1
    mean = train_x.float().mean(0) if center else 0
    train, preds = F.normalize(train_x.float() - mean, dim=1), []
    for chunk in F.normalize(test_x.float() - mean, dim=1).split(1024):
        sim, idx = (chunk @ train.T).topk(min(k, len(train)), dim=1)
        votes = torch.zeros(len(chunk), num_classes, device=chunk.device)
        preds.append(votes.scatter_add_(1, train_y[idx], (sim / temperature).exp()).argmax(1))
    pred = torch.cat(preds)
    return Probe((pred == test_y).float().mean().item(), pred)


class AttentiveProbe(nn.Module):
    """A learned query token cross-attends to frozen tokens ``(B, N, D)``, followed by an MLP and a linear classifier.

    Bardes et al., "Revisiting Feature Prediction for Learning Visual Representations from Video" (V-JEPA), 2024.
    Paper: https://arxiv.org/abs/2404.08471
    Code: https://github.com/facebookresearch/jepa
    """

    def __init__(self, dim: int, num_classes: int, num_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.norm_x, self.norm_q, self.norm_mlp, self.norm_out = (nn.LayerNorm(dim) for _ in range(4))
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        q, x = self.query.expand(len(x), -1, -1), self.norm_x(x)
        q = q + self.attn(self.norm_q(q), x, x, need_weights=False)[0]
        q = q + self.mlp(self.norm_mlp(q))
        return self.head(self.norm_out(q[:, 0]))


def attentive_probe(
    train_x: Tensor,
    train_y: Tensor,
    test_x: Tensor,
    test_y: Tensor,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    num_heads: int = 8,
    generator: torch.Generator | None = None,
) -> Probe:
    """Train an ``AttentiveProbe`` on frozen tokens ``(N, T, D)`` with class labels ``(N,)`` and report test accuracy,
    the evaluation of V-JEPA and V-JEPA 2. Unlike a linear probe on pooled features, it can read which tokens
    matter."""
    num_classes = int(max(train_y.max(), test_y.max())) + 1
    probe = AttentiveProbe(train_x.size(-1), num_classes, num_heads).to(train_x.device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * -(-len(train_x) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    for _ in range(epochs):
        order = torch.randperm(len(train_x), generator=generator).to(train_x.device)
        for idx in order.split(batch_size):
            loss = F.cross_entropy(probe(train_x[idx].float()), train_y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    probe.eval()
    with torch.no_grad():
        pred = torch.cat([probe(chunk.float()).argmax(1) for chunk in test_x.split(batch_size)])
    return Probe((pred == test_y).float().mean().item(), pred)
