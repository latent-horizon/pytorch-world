import pytest
import torch

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from torch_world import plots
from torch_world.envs import PointMass


def test_figures_render():
    images = PointMass(batch_size=4, size=16).reset()
    paths, goals = torch.rand(4, 6, 2), torch.rand(4, 2)
    figures = [
        plots.images(images, ["a"] * 4, ncols=2),
        plots.masks(images[0], (4, 4), {"context": torch.arange(8), "target": torch.tensor([12, 13])}, title="masks"),
        plots.curves({"loss": torch.rand(50).add(0.1).tolist(), "sparse": ([0, 5, 9], [1.0, 0.5, 0.2])}),
        plots.trajectories(paths, goals, background=images),
        plots.distances({"plan": torch.rand(4, 6)}, radius=0.1),
    ]
    for fig in figures:
        assert isinstance(fig, matplotlib.figure.Figure) and fig.axes


def test_result_figures_render():
    images = (torch.rand(3, 3, 8, 8) * 255).to(torch.uint8)
    figures = [
        plots.images(images, rows=["row"], ncols=3),
        plots.bars({"a": {"model": 0.9, "baseline": 0.5}, "b": {"model": 0.7, "baseline": 0.6}}, baseline="baseline"),
        plots.answers(images, ["q ?"] * 3, ["yes", "no", "yes"], ["yes", "yes", "yes"], ncols=3),
        plots.probes(torch.rand(20, 2), torch.rand(20, 2), ["x", "y"]),
    ]
    for fig in figures:
        assert isinstance(fig, matplotlib.figure.Figure) and fig.axes


def test_episodes_animation():
    frames = (torch.rand(3, 4, 3, 8, 8) * 255).to(torch.uint8)
    gif = plots.episodes(frames, success=[True, False, True], titles=["a", "b", "c"], ncols=3)
    assert gif.data[:6] == b"GIF89a" and "image/gif;base64" in gif._repr_html_()
