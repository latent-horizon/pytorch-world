import math

import pytest
import torch

from torch_world.metrics import (
    attentive_probe,
    chamfer_distance,
    first_success,
    knn_accuracy,
    lidar,
    linear_probe,
    path_length,
    progress,
    psnr,
    r2_score,
    rankme,
    recall_at_k,
    rollout_error,
    smoothness,
    spl,
    ssim,
    success_rate,
    topk_accuracy,
)


def test_rankme_counts_used_dimensions():
    g = torch.Generator().manual_seed(0)
    full = torch.linalg.qr(torch.randn(256, 16, generator=g))[0]
    assert rankme(full) == pytest.approx(16, rel=1e-4)
    collapsed = torch.randn(256, 1, generator=g) * torch.randn(1, 16, generator=g)
    assert rankme(collapsed) == pytest.approx(1, abs=1e-3)
    shared = full + 100 * torch.ones(16)
    assert rankme(shared) < 1.1 and rankme(shared, center=True) == pytest.approx(rankme(full - full.mean(0)), rel=1e-6)


def test_lidar_counts_discriminative_dimensions():
    g = torch.Generator().manual_seed(0)
    means = torch.cat([torch.randn(4000, 3, generator=g), torch.zeros(4000, 5)], 1)
    views = means[:, None] + 0.01 * torch.randn(4000, 2, 8, generator=g)
    assert lidar(views) == pytest.approx(3, abs=0.05)
    assert lidar(views[..., [0, 3, 4, 5, 6, 7]]) == pytest.approx(1, abs=0.02)


def test_linear_probe_regression_and_classes():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(600, 10, generator=g)
    y = x[:, :3] @ torch.randn(3, 2, generator=g) + 1.0
    probe = linear_probe(x[:500], y[:500], x[500:], y[500:], ridge=1e-6)
    assert probe.score == pytest.approx(1, abs=1e-4) and probe.predictions.shape == (100, 2)
    labels = (x[:, 0] > 0).long() + 2 * (x[:, 1] > 0).long()
    assert linear_probe(x[:500], labels[:500], x[500:], labels[500:]).score > 0.9
    assert r2_score(y[:, 0], y[:, 0]) == 1.0 and r2_score(torch.zeros(4), torch.arange(4.0)) < 0


def test_knn_accuracy_on_separated_clusters():
    g = torch.Generator().manual_seed(0)
    labels = torch.arange(300) % 3
    x = torch.eye(3)[labels] * 5 + 0.3 * torch.randn(300, 3, generator=g)
    result = knn_accuracy(x[:240], labels[:240], x[240:], labels[240:], k=10)
    assert result.score == 1.0 and torch.equal(result.predictions, labels[240:])
    shifted = x + 100
    assert knn_accuracy(shifted[:240], labels[:240], shifted[240:], labels[240:], k=10, center=True).score == 1.0


def test_attentive_probe_finds_the_informative_token():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2400, 6, 16, generator=g)
    marked = torch.randint(6, (2400,), generator=g)
    x[torch.arange(2400), marked, 1] = 4.0
    labels = (x[torch.arange(2400), marked, 0] > 0).long()
    split = (x[:2000], labels[:2000], x[2000:], labels[2000:])
    attentive = attentive_probe(*split, epochs=30, lr=3e-3, num_heads=4, generator=g).score
    pooled = linear_probe(split[0].mean(1), split[1], split[2].mean(1), split[3]).score
    assert attentive > 0.85 and attentive > pooled + 0.15


def test_rollout_error_per_step():
    target = torch.zeros(2, 3, 4)
    pred = torch.stack([torch.zeros(2, 4), torch.ones(2, 4), 2 * torch.ones(2, 4)], 1)
    assert rollout_error(pred, target).tolist() == [0, 1, 2]
    assert rollout_error(pred, target, p=2).tolist() == [0, 1, 4]


def test_psnr_and_ssim():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(2, 3, 3, 32, 32, generator=g)
    assert psnr(x + 0.1, x).shape == (2, 3) and torch.allclose(psnr(x + 0.1, x), torch.tensor(20.0))
    assert torch.allclose(ssim(x, x), torch.ones(2, 3))
    noisy = (x + 0.2 * torch.randn(x.shape, generator=g)).clamp(0, 1)
    assert (ssim(noisy, x) < 0.9).all() and (ssim(noisy, x) > ssim(torch.rand(x.shape, generator=g), x)).all()


def test_chamfer_distance():
    a = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    b = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]])
    assert chamfer_distance(a, a).item() == 0
    assert chamfer_distance(a, b).item() == pytest.approx(1 / 3)


def test_retrieval_metrics():
    sim = torch.tensor([[0.9, 0.1, 0.0], [0.8, 0.5, 0.0], [0.1, 0.2, 0.3]])
    assert recall_at_k(sim, 1) == pytest.approx(2 / 3)
    assert recall_at_k(sim, (1, 2)) == {"R@1": pytest.approx(2 / 3), "R@2": 1.0}
    assert recall_at_k(sim, 1, targets=torch.tensor([0, 0, 2])) == 1.0
    assert topk_accuracy(sim, torch.tensor([0, 1, 1]), k=1) == pytest.approx(1 / 3)
    assert topk_accuracy(sim, torch.tensor([0, 1, 1]), k=2) == 1.0
    assert not math.isnan(recall_at_k(torch.zeros(4, 4), 1))


def test_success_rate_wilson_interval():
    rate = success_rate(torch.tensor([True] * 8 + [False] * 8))
    assert rate.value == 0.5 and rate.n == 16
    assert rate.low == pytest.approx(0.2800, abs=1e-3) and rate.high == pytest.approx(0.7200, abs=1e-3)
    none = success_rate(torch.zeros(10, dtype=torch.bool))
    assert none.low == 0 and none.high == pytest.approx(0.2775, abs=1e-3) and str(none) == "0% [0%, 28%]"


def test_episode_measures():
    reached = torch.tensor([[False, False, True, True], [False, False, False, False]])
    assert first_success(reached).tolist() == [2, -1]
    assert progress(torch.tensor([[2.0, 1.0, 0.5], [1.0, 1.0, 2.0]])).tolist() == [0.75, -1.0]
    assert progress(torch.tensor([[0.0, 0.1]])).isnan().all()  # undefined when starting at the goal
    straight = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    detour = torch.tensor([[[0.0, 0.0], [0.0, 1.0], [2.0, 1.0], [2.0, 0.0]]])
    assert path_length(detour).item() == 4
    assert spl(torch.tensor([True]), straight, torch.tensor([[2.0, 0.0]])) == 1.0
    assert spl(torch.tensor([True]), detour, torch.tensor([[2.0, 0.0]])) == 0.5
    assert spl(torch.tensor([False]), straight, torch.tensor([[2.0, 0.0]])) == 0.0
    steady, jerky = torch.ones(1, 4, 2), torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]]])
    assert smoothness(steady).item() == 0 and smoothness(jerky).item() == 2
