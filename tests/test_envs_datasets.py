import torch

from torch_world.datasets import TrajectoryDataset
from torch_world.envs import PointMass
from torch_world.utils import ema_update, linear_schedule


def test_point_mass_dynamics_and_render():
    env = PointMass(batch_size=2, size=16, max_step=0.1)
    env.reset(torch.tensor([[0.5, 0.53], [0.95, 0.05]]))
    pixels = env.step(torch.tensor([[1.0, 0.0], [5.0, -5.0]]))
    assert torch.allclose(env.state, torch.tensor([[0.6, 0.53], [1.0, 0.0]]))
    assert pixels.shape == (2, 3, 16, 16) and pixels.min() >= 0 and pixels.max() <= 1
    darkest = pixels[0].sum(0).flatten().argmin()
    assert divmod(int(darkest), 16) == (8, 9)


def test_collect_and_windows():
    data = PointMass(batch_size=3, size=8).collect(6, generator=torch.Generator().manual_seed(0))
    assert data["pixels"].shape == (3, 6, 3, 8, 8) and data["actions"].shape == (3, 6, 2)
    step = (data["states"][:, 1:] - data["states"][:, :-1]).abs()
    assert (step <= 0.1 + 1e-6).all()
    dataset = TrajectoryDataset(data, window=4)
    assert len(dataset) == 9
    assert torch.equal(dataset[4]["states"], data["states"][1, 1:5])


def test_windows_with_frameskip_stack_actions():
    data = {"states": torch.arange(10.0).view(1, 10, 1), "actions": torch.arange(20.0).view(1, 10, 2)}
    dataset = TrajectoryDataset(data, window=2, frameskip=3)
    assert len(dataset) == 5
    window = dataset[1]
    assert window["states"].flatten().tolist() == [1, 4]
    assert window["actions"].tolist() == [[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13]]


def test_ema_and_schedule():
    a, b = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    expected = 0.5 * a.weight + 0.5 * b.weight
    ema_update(a, b, 0.5)
    assert torch.allclose(a.weight, expected)
    assert linear_schedule(0.996, 1.0, 50, 100) == 0.998 and linear_schedule(0, 1, 200, 100) == 1


def test_recompute_batchnorm_matches_data_statistics():
    from torch_world.utils import recompute_batchnorm

    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3)).eval()
    data = [torch.randn(64, 3) * 2 + 1 for _ in range(4)]
    recompute_batchnorm(model, data, model)
    with torch.no_grad():
        h = model[0](torch.cat(data))
    assert torch.allclose(model[1].running_mean, h.mean(0), atol=1e-5)
    assert not model.training and model[1].momentum == 0.1


def test_sample_goals_and_run_policy():
    from torch_world.envs import run_policy

    env = PointMass(batch_size=64, size=8)
    env.reset(generator=torch.Generator().manual_seed(0))
    goals = env.sample_goals((0.2, 0.3))
    d = (goals - env.state).norm(dim=-1)
    assert ((d >= 0.2 - 1e-6) & (d <= 0.3 + 1e-6)).all() and ((goals >= 0) & (goals <= 1)).all()
    states = run_policy(env, lambda obs: torch.ones(len(obs), 2), steps=3)
    assert states.shape == (64, 4, 2) and torch.equal(states[:, -1], env.state)


def test_macaw_flies_in_bounds_and_flaps():
    from torch_world.envs import Macaw

    env = Macaw(batch_size=2, size=32)
    env.reset(torch.tensor([[0.5, 0.5], [0.8, 0.2]]))
    first = env.render()
    pixels = env.step(torch.tensor([[0.0, 0.0], [1.0, -1.0]]))
    assert pixels.shape == (2, 3, 32, 32) and pixels.min() >= 0 and pixels.max() <= 1
    assert torch.allclose(env.state, torch.tensor([[0.5, 0.5], [0.85, 0.15]]))
    assert not torch.allclose(pixels[0], first[0])
    assert torch.allclose(env.render(env.state)[0], first[0])
    data = env.collect(5, generator=torch.Generator().manual_seed(0))
    assert ((data["states"] >= Macaw.low) & (data["states"] <= Macaw.high)).all()
