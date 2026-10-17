"""Original environments retain their APIs while benchmark recipes share one protocol."""

import pytest
import torch

from torch_world.envs import BenchmarkEnv, Macaw, PointMass, get_spec, list_envs, make


@pytest.mark.parametrize("name", ["PointMass", "Macaw", "PushT", "RobotCube", "RobotCubeSO101"])
def test_legacy_reference_and_episode_contract(name):
    if name not in {"PointMass", "Macaw"}:
        pytest.importorskip("mujoco")
    with make(name + "-v0", observation="state", size=8, render_size=24, quality="low") as world:
        world.reset(seed=7)
        start, goal = world.state.clone(), world.goal.clone()
        assert not world.success(start, goal).any()
        world.reset(seed=19)
        assert torch.equal(world.goal, goal)  # MPC encodes one fixed target for all evaluation seeds.
        world.reset(seed=7)
        assert torch.equal(world.state, start)
        world.render_goal_frame()
        assert torch.equal(world.state, start)
        policy = world.reference_policy()
        peak = 0
        for _ in range(world.max_episode_steps):
            world.step(policy(None))
            if name.startswith("RobotCube"):
                peak = max(peak, world.state[0, 6].item())
            if world.transition.success.all():
                break
        assert world.transition.success.all()
        if name.startswith("RobotCube"):
            assert peak > world.world.half + 0.02
            assert world.state[0, 3] > 0.9 and world.state[0, 2] > 0.8 * world.world.hover


def test_macaw_presentation_preserves_wing_phase_and_legacy_api():
    assert not isinstance(Macaw(size=8), BenchmarkEnv)
    assert not isinstance(PointMass(size=8), BenchmarkEnv)
    assert len(list_envs(tag="legacy")) == 5
    with make("Macaw-v0", size=16, render_size=48) as world:
        world.reset(seed=7)
        first = world.render_frame()
        world.step(torch.zeros(1, 2))
        phase, state = world.world.phase.clone(), world.state.clone()
        second = world.render_frame()
        world.render_goal_frame()
        assert first.shape == second.shape == (1, 3, 48, 48)
        assert not torch.equal(first, second)
        assert torch.equal(phase, world.world.phase) and torch.equal(state, world.state)


@pytest.mark.parametrize("name", ["PointMass", "Macaw", "PushT"])
def test_legacy_aligned_collection_and_absorbing_timeout(name):
    if name == "PushT":
        pytest.importorskip("mujoco")
    with make(name + "-v0", size=8, max_episode_steps=2) as world:
        data = world.collect(4, generator=torch.Generator().manual_seed(7))
        assert data["valid"].tolist() == [[True, True, False, False]]
        assert torch.equal(data["next_observations"][:, 0], data["observations"][:, 1])
        assert not data["actions"][:, 2:].any() and not data["rewards"][:, 2:].any()
        final = world.state.clone()
        world.step(torch.ones(1, world.action_dim))
        assert torch.equal(world.state, final)


@pytest.mark.parametrize("name", ["PointMass", "Macaw", "RobotCube"])
def test_named_world_uses_shared_experiment_and_artifacts(tmp_path, name):
    from torch_world import envs
    from torch_world.benchmarks import load_run, run_experiment

    if name == "RobotCube":
        pytest.importorskip("mujoco")
    with getattr(envs, name)(size=8, render_size=24, quality="low") as world:
        result = run_experiment(
            world,
            samples=16,
            updates=1,
            evaluation_seeds=1,
            evaluation_steps=3,
            output_dir=tmp_path,
            save_gifs=False,
        )
        saved = load_run(result["run_dir"])
        assert saved["env_id"] == name + "-v0"
        assert set(saved["evaluation"]) == {"random", "LeWM + CEM"}
        assert saved["config"]["environment_settings"]["environment"] == name
        assert world.as_benchmark().world is world
        assert world.reset(generator=torch.Generator().manual_seed(7)).shape == (1, 3, 8, 8)
        replay_settings = dict(world.as_benchmark().settings)
        with make(saved["env_id"], **replay_settings) as replay:
            assert torch.equal(replay.reset(seed=7), world.as_benchmark().reset(seed=7))


@pytest.mark.parametrize(
    "name,options",
    [
        ("PointMass", {"sigma": 0.08}),
        ("Macaw", {"scale": 0.9}),
        ("PushT", {}),
        ("RobotCube", {}),
        ("RobotCube", {"robot": "so101"}),
    ],
)
def test_named_world_reference_gif_and_original_collection(tmp_path, name, options):
    from torch_world import envs
    from torch_world.benchmarks.visualization import show_env
    from torch_world.envs.registry import environment_scope

    capture_output = pytest.importorskip("IPython.utils.capture").capture_output
    Image = pytest.importorskip("PIL.Image")
    pytest.importorskip("matplotlib")
    if name in {"PushT", "RobotCube"}:
        pytest.importorskip("mujoco")
    cls = getattr(envs, name)
    with cls(size=8, render_size=24, **options) as world:
        assert world.collect(2, generator=torch.Generator().manual_seed(7))["pixels"].shape == (1, 2, 3, 8, 8)
        benchmark = world.as_benchmark()
        assert benchmark is world.as_benchmark() and benchmark.world is world
        expected = "RobotCubeSO101" if options.get("robot") == "so101" else name
        assert get_spec(world).id == expected + "-v0"
        with environment_scope(world) as borrowed:
            assert borrowed is benchmark
        with pytest.raises(ValueError, match="constructor"), environment_scope(world, size=16):
            pass
        path = tmp_path / "success.gif"
        with capture_output():
            show_env(world)
            world.show_env()
            world.show_success(path=path)
        with Image.open(path) as gif:
            assert gif.n_frames > 1
        assert benchmark.transition.success.all()
        assert benchmark.render_settings["render_size"] == 24
    with make(cls, size=8, render_size=24, **options) as constructed:
        assert get_spec(constructed).id == expected + "-v0"


def test_class_experiment_records_selected_robot(tmp_path):
    pytest.importorskip("mujoco")
    from torch_world.benchmarks import load_run, run_experiment
    from torch_world.envs import RobotCube

    result = run_experiment(
        RobotCube,
        env_kwargs={"robot": "so101", "size": 8, "render_size": 24, "quality": "low"},
        samples=8,
        updates=0,
        evaluation_seeds=1,
        evaluation_steps=1,
        save_gifs=False,
        output_dir=tmp_path,
    )
    saved = load_run(result["run_dir"])
    assert saved["env_id"] == saved["config"]["environment"] == "RobotCubeSO101-v0"
    assert saved["config"]["environment_settings"]["options"]["robot"] == "so101"


def test_legacy_quality_controls_rendering_without_changing_physics():
    pytest.importorskip("mujoco")
    with make("PushT-v0", quality="low", size=8) as low, make("PushT-v0", quality="high", size=8) as high:
        low.reset(seed=7)
        high.reset(seed=7)
        for world in (low, high):
            world.step(torch.tensor([[0.0, -0.5]]))
        assert torch.equal(low.state, high.state)
        assert not low.world.render_shadows and high.world.render_shadows
