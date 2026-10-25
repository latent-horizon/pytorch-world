import math

import torch

from torch_world.metrics import crps, energy_score, gaussian_nll, interval_coverage, spread_skill


def brute_crps(samples, target, fair=False):
    S = samples.size(1)
    skill = (samples - target[:, None]).abs().mean(1)
    pairs = (samples[:, :, None] - samples[:, None]).abs().sum((1, 2))
    return skill - 0.5 * pairs / (S * (S - 1) if fair else S * S)


def test_crps_matches_pairwise_definition_and_point_forecast():
    g = torch.Generator().manual_seed(0)
    samples, target = torch.randn(4, 7, 3, generator=g), torch.randn(4, 3, generator=g)
    assert crps(samples, target).shape == (4, 3)
    torch.testing.assert_close(crps(samples, target), brute_crps(samples, target))
    torch.testing.assert_close(crps(samples, target, fair=True), brute_crps(samples, target, fair=True))
    torch.testing.assert_close(crps(samples[:, :1], target), (samples[:, 0] - target).abs())


def test_crps_of_gaussian_ensemble_matches_closed_form():
    g = torch.Generator().manual_seed(0)
    samples, y = torch.randn(1, 20000, 1, generator=g), torch.tensor([[0.5]])
    pdf, cdf = math.exp(-0.125) / math.sqrt(2 * math.pi), 0.5 * (1 + math.erf(0.5 / math.sqrt(2)))
    exact = 0.5 * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi)
    assert abs(crps(samples, y, fair=True).item() - exact) < 0.01


def test_energy_score_reduces_to_crps_in_one_dimension_and_to_error_for_points():
    g = torch.Generator().manual_seed(0)
    samples, target = torch.randn(5, 6, 2, 1, generator=g), torch.randn(5, 2, 1, generator=g)
    score = energy_score(samples, target)
    assert score.shape == (5, 2)
    torch.testing.assert_close(score, crps(samples, target)[..., 0])
    vectors, goal = torch.randn(5, 1, 2, 4, generator=g), torch.randn(5, 2, 4, generator=g)
    torch.testing.assert_close(energy_score(vectors, goal), (vectors[:, 0] - goal).norm(dim=-1))
    torch.testing.assert_close(
        energy_score(vectors, goal, event_dims=2), (vectors[:, 0] - goal).flatten(1).norm(dim=-1)
    )


def test_energy_score_prefers_the_true_distribution():
    g = torch.Generator().manual_seed(0)
    target = torch.randn(2000, 3, generator=g)
    right, narrow = torch.randn(2000, 16, 3, generator=g), 0.2 * torch.randn(2000, 16, 3, generator=g)
    assert energy_score(right, target).mean() < energy_score(narrow, target).mean()


def test_gaussian_nll_matches_torch():
    g = torch.Generator().manual_seed(0)
    mean, var, target = (
        torch.randn(4, 3, generator=g),
        torch.rand(4, 3, generator=g) + 0.1,
        torch.randn(4, 3, generator=g),
    )
    expected = torch.nn.functional.gaussian_nll_loss(mean, target, var, full=True, reduction="none")
    torch.testing.assert_close(gaussian_nll(mean, var, target), expected)


def test_interval_coverage_is_nominal_for_a_calibrated_ensemble():
    g = torch.Generator().manual_seed(0)
    samples, target = torch.randn(4000, 64, 2, generator=g), torch.randn(4000, 2, generator=g)
    levels = (0.5, 0.8, 0.9)
    cover = interval_coverage(samples, target, levels)
    assert cover.shape == (3, 4000, 2)
    for level, c in zip(levels, cover.mean((1, 2))):
        assert abs(c.item() - level) < 0.04
    assert interval_coverage(0.3 * samples, target, levels).mean((1, 2))[0] < 0.3  # overconfident
    assert interval_coverage(samples[:, :1], target).sum() == 0  # a point covers nothing


def test_spread_skill_ratio_near_one_when_calibrated_and_zero_for_points():
    g = torch.Generator().manual_seed(0)
    center = torch.randn(3000, 8, generator=g)
    samples, target = (
        center[:, None] + torch.randn(3000, 16, 8, generator=g),
        center + torch.randn(3000, 8, generator=g),
    )
    spread, skill = spread_skill(samples, target)
    assert abs(spread.item() / skill.item() - 1) < 0.05
    spread, skill = spread_skill(samples, target, dim=0)
    assert spread.shape == skill.shape == (8,)
    spread, skill = spread_skill(samples[:, :1], target)
    assert spread == 0 and skill > 0
