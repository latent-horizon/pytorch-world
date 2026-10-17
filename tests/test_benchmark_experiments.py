"""Model interchangeability and procedural artifact persistence."""

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from torch_world.benchmarks import load_run, run_experiment, save_run, summarize_runs, write_summary
from torch_world.models.base import WorldModel
from torch_world.planning import CEM


class AlternativeModel(WorldModel):
    history = 1

    def __init__(self, shape, action_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(torch.tensor(shape).prod().item(), 8))
        self.dynamics = nn.Linear(8 + action_dim, 8)

    def encode(self, observations):
        return self.encoder(observations.flatten(0, 1)).unflatten(0, observations.shape[:2])

    def predict(self, z, actions):
        return self.dynamics(torch.cat([z, actions], -1))

    def forward(self, observations, actions):
        z = self.encode(observations)
        return {"loss": (self.predict(z[:, :1], actions[:, :1]) - z[:, 1:]).square().mean()}


def small_planner(action_dim):
    return CEM(1, action_dim, num_samples=4, num_elites=2, iterations=1, bounds=(-1, 1))


def test_model_factory_artifact_roundtrip_and_summary(tmp_path):
    result = run_experiment(
        "Maze-v0",
        model_factory=AlternativeModel,
        planner_factory=small_planner,
        updates=1,
        samples=16,
        evaluation_seeds=1,
        evaluation_steps=2,
        env_kwargs={"size": 8, "render_size": 16},
        output_dir=tmp_path,
        save_gifs=False,
        device="cpu",
    )
    assert result["model"] == "AlternativeModel"
    assert "AlternativeModel + CEM" in result["evaluation"]
    folder = Path(result["run_dir"])
    config = json.loads((folder / "config.json").read_text())
    assert config["environment_settings"]["size"] == 8
    assert config["render_settings"] == {"quality": "high", "observation_size": 8, "render_size": 16}
    assert config["planner_settings"]["horizon"] == 1
    assert config["model_factory"].endswith(":AlternativeModel")
    episodes = torch.load(folder / "episodes.pt", weights_only=True)
    assert episodes["random"].shape == (3, 3, 16, 16)
    loaded = load_run(folder)
    assert loaded["evaluation"] == result["evaluation"]
    assert torch.equal(loaded["target"], result["target"])
    summaries = summarize_runs(tmp_path)
    assert len(summaries) == 2
    assert summaries[0]["quality"] == "high" and summaries[0]["render_size"] == 16
    second = save_run(result, tmp_path, gifs=False)
    assert second != folder
    # Interrupted directories never appear in comparisons.
    (tmp_path / "interrupted").mkdir()
    (tmp_path / "interrupted" / "metrics.json").write_text("{}")
    rows = write_summary(tmp_path, tmp_path / "summary.csv")
    assert len(rows) == 4
    assert (tmp_path / "summary.csv").read_text().count("AlternativeModel + CEM") == 2
    (folder / "metrics.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        summarize_runs(tmp_path)


def test_state_planning_uses_actual_simulator_returns():
    pytest.importorskip("mujoco")
    result = run_experiment(
        "CartpoleSwingup-v0",
        model_factory=AlternativeModel,
        planner_factory=small_planner,
        updates=1,
        samples=32,
        evaluation_seeds=1,
        evaluation_steps=2,
        env_kwargs={"observation": "state"},
        device="cpu",
    )
    assert result["observation"] == "state"
    assert all("final_distances" in row and "success_rate" in row for row in result["evaluation"].values())


def test_success_display_verifies_policy_and_labels_gt(monkeypatch):
    from IPython import display

    from torch_world.benchmarks import show_success

    shown = []
    monkeypatch.setattr(display, "display", shown.append)
    animation = show_success("KeyDoor-v0", env_kwargs={"size": 8})
    assert animation is not None
    assert any("verified reference" in str(getattr(item, "data", "")) for item in shown)
    with pytest.raises(ValueError, match="did not produce a verified success"):
        show_success("Maze-v0", policy=lambda obs: torch.zeros(len(obs), 2), steps=2)


def test_saved_gif_contains_gt_and_target_roundtrips(tmp_path):
    from PIL import Image

    result = run_experiment(
        "Maze-v0",
        model_factory=AlternativeModel,
        planner_factory=small_planner,
        updates=1,
        samples=8,
        evaluation_seeds=1,
        evaluation_steps=1,
        env_kwargs={"size": 8},
        output_dir=tmp_path,
    )
    folder = Path(result["run_dir"])
    target = torch.load(folder / "target.pt", weights_only=True)
    assert torch.equal(target, result["target"])
    gif = Image.open(folder / "random.gif")
    # Two fixed panels: GT and the recorded rollout.
    assert gif.width > gif.height * 1.5


def test_named_environment_preview_and_experiment(monkeypatch):
    from IPython import display

    from torch_world.envs import KeyDoor

    shown = []
    monkeypatch.setattr(display, "display", shown.append)
    world = KeyDoor(size=8)
    assert world.show_env() is None
    figure = shown[-1]
    assert [ax.get_title(loc="center") for ax in figure.axes] == ["Initial", "GT"]
    assert any("KeyDoor-v0" in str(getattr(item, "data", "")) for item in shown)
    settings = {
        "model_factory": AlternativeModel,
        "planner_factory": small_planner,
        "updates": 1,
        "samples": 16,
        "evaluation_seeds": 1,
        "evaluation_steps": 2,
        "env_kwargs": {"size": 8},
        "device": "cpu",
    }
    named = run_experiment(KeyDoor, **settings)
    registered = run_experiment("KeyDoor-v0", **settings)
    assert named["evaluation"] == registered["evaluation"]
    assert named["heldout_latent_mse"] == registered["heldout_latent_mse"]
    assert named["config"]["environment"] == "KeyDoor-v0"
    assert named["config"]["environment_settings"]["task"] == "key_door"


def test_instance_lifecycle_settings_and_preview_reuse(monkeypatch):
    from IPython import display

    from torch_world.envs import Maze

    class TrackedMaze(Maze):
        created = 0

        def __init__(self, **kwargs):
            type(self).created += 1
            self.closed = False
            super().__init__(**kwargs)

        def close(self):
            self.closed = True

    monkeypatch.setattr(display, "display", lambda value: None)
    level = ("#######", "#A...G#", "#######")
    world = TrackedMaze(level=level, size=12, batch_size=2)
    world.show_env()
    settings = {
        "model_factory": AlternativeModel,
        "planner_factory": small_planner,
        "updates": 1,
        "samples": 16,
        "evaluation_seeds": 1,
        "evaluation_steps": 2,
        "device": "cpu",
    }
    result = run_experiment(world, **settings)
    assert result["config"]["environment_settings"]["level"] == list(level)
    assert result["config"]["environment_settings"]["size"] == 12
    assert result["config"]["environment_settings"]["batch_size"] == 2
    assert len(result["evaluation"]["random"]["successes"]) == 2
    world.show_success()
    assert TrackedMaze.created == 1 and not world.closed
    assert world.transition.success.all()
    with pytest.raises(ValueError, match="constructor"):
        run_experiment(world, env_kwargs={"size": 8}, **settings)
    assert not world.closed

    def broken_model(*args):
        raise RuntimeError("model initialization failed")

    with pytest.raises(RuntimeError, match="model initialization"):
        run_experiment(world, model_factory=broken_model, samples=16, updates=1)
    assert not world.closed
    world.close()
    assert world.closed


def test_partial_factory_roundtrip(tmp_path):
    from functools import partial

    result = run_experiment(
        "Maze-v0",
        model_factory=partial(AlternativeModel),
        planner_factory=partial(CEM, 1, num_samples=4, num_elites=2, iterations=1),
        samples=8,
        updates=1,
        evaluation_seeds=1,
        evaluation_steps=1,
        env_kwargs={"size": 8},
        output_dir=tmp_path,
        save_gifs=False,
        device="cpu",
    )
    config = load_run(result["run_dir"])["config"]
    assert config["model_factory"]["factory"].endswith(":AlternativeModel")
    assert config["planner_factory"]["args"] == [1]


def test_tiny_puzzle_split_has_actionable_error():
    from torch_world.envs import Maze

    with (
        Maze(level=("####", "#AG#", "####")) as world,
        pytest.raises(ValueError, match="two reachable nonterminal states"),
    ):
        run_experiment(world, updates=0, samples=2)


@pytest.mark.parametrize("length", [1, 2, 3, 4, 60, 61])
def test_episode_subsampling_preserves_final_observation_and_clock(monkeypatch, length):
    from torch_world import plots
    from torch_world.benchmarks.visualization import episode_gif

    captured = {}

    def capture(frames, **kwargs):
        captured.update(frames=frames, **kwargs)

    monkeypatch.setattr(plots, "episodes", capture)
    frames = torch.arange(length, dtype=torch.float32)[:, None, None, None].expand(-1, 3, 4, 4)
    target = torch.full((3, 4, 4), -1.0)
    episode_gif(frames, target, title="policy", stride=2)
    assert torch.equal(captured["frames"][1, -1], frames[-1])
    assert torch.equal(captured["frames"][0], target.expand_as(captured["frames"][0]))
    assert captured["frame_steps"][0] == 0
    assert captured["frame_steps"][-1] == length - 1
    assert captured["frame_steps"] == captured["frames"][1, :, 0, 0, 0].tolist()


class RewardOnlyWorld:
    """Reuse the observation buffer to expose snapshot aliasing in evaluators."""

    batch_size, max_episode_steps, device = 1, 2, torch.device("cpu")

    def reset(self, seed):
        from torch_world.envs import Transition

        self.obs = torch.zeros(1, 2)
        self.transition = Transition(torch.zeros(1), torch.tensor([False]), torch.tensor([False]))
        return self.obs

    def step(self, action):
        self.obs += 1
        self.transition.reward = torch.ones(1)
        self.transition.truncated = self.obs[:, 0] >= self.max_episode_steps
        return self.obs


def test_reward_only_evaluation_does_not_invent_failure_and_snapshots_frames():
    from torch_world.benchmarks.experiment import _evaluate_policy

    scores, frames = _evaluate_policy(
        RewardOnlyWorld(),
        lambda obs: torch.zeros(1, 1),
        seeds=[7],
        steps=2,
        goal_scoring=False,
    )
    assert "successes" not in scores and "success_rate" not in scores
    assert scores["returns"] == [2.0]
    assert frames[:, 0].tolist() == [0, 1, 2]


def test_success_display_rejects_nonpositive_budget():
    from torch_world.benchmarks import show_success

    with pytest.raises(ValueError, match="steps must be positive"):
        show_success("Maze-v0", steps=0)


def test_reward_planning_records_effective_objective():
    from torch_world.envs import BenchmarkEnv, EnvSpec, Transition, register_env

    class RewardTask(BenchmarkEnv):
        env_id, action_dim = "TestRewardOnly-v0", 1

        def __init__(self):
            super().__init__(max_episode_steps=2)
            self.settings = {}

        def reset(self, generator=None, seed=None):
            self._reset_episode()
            self.state = torch.zeros(1, 2)
            return self.state.clone()

        def step(self, action):
            self.state = self.state + self._action(action)
            self.elapsed += 1
            self.transition = Transition(
                self.state[:, 0].clone(),
                torch.tensor([False]),
                self.elapsed >= 2,
            )
            return self.state.clone()

        def render_target(self):
            return torch.zeros_like(self.state)

    register_env(EnvSpec(RewardTask.env_id, RewardTask, "test", "Reward-only test task"))
    with RewardTask() as world:
        result = run_experiment(
            world,
            model_factory=AlternativeModel,
            planner_factory=small_planner,
            samples=8,
            updates=1,
            evaluation_seeds=1,
            evaluation_steps=2,
            device="cpu",
        )
    assert result["config"]["planner_settings"]["objective"] == "path"
    assert all("success_rate" not in scores for scores in result["evaluation"].values())


def test_offline_puzzle_controls_match_online_dynamics_including_waits(monkeypatch):
    from torch_world.benchmarks.experiment import _puzzle_data
    from torch_world.envs import Maze

    with Maze(size=8) as world:
        starts = []
        render = world.render

        def capture(state):
            starts.append(state.clone())
            return render(state)

        monkeypatch.setattr(world, "render", capture)
        (observations, actions), _ = _puzzle_data(world, 128, torch.Generator().manual_seed(7))
    assert (actions[:, 0].abs().amax(-1) < 0.5).any()
    assert ((actions[:, 0].abs() > 0) & (actions[:, 0].abs() < 1)).all()
    with Maze(batch_size=128, size=8) as batched:
        assert torch.equal(observations[:, 0], batched.reset(starts[0]))
        assert torch.equal(observations[:, 1], batched.step(actions[:, 0]))


def test_success_is_animated_saved_and_preserves_limit(tmp_path, monkeypatch):
    import io

    from IPython import display
    from PIL import Image

    from torch_world.benchmarks import show_success
    from torch_world.envs import Maze
    from torch_world.plots import Animation

    monkeypatch.setattr(display, "display", lambda value: None)
    world = Maze(quality="low", max_episode_steps=2)
    path = tmp_path / "reference" / "episode.gif"
    animation = show_success(world, path=path)
    assert isinstance(animation, Animation)
    assert path.read_bytes() == animation.data
    with Image.open(io.BytesIO(animation.data)) as gif:
        assert gif.is_animated and gif.n_frames > 2
        gif.seek(0)
        first = gif.convert("RGB").tobytes()
        gif.seek(gif.n_frames - 1)
        assert first != gif.convert("RGB").tobytes()
    assert world.transition.success[0]
    assert world.max_episode_steps == 2
    world.close()


def test_missing_success_reference_does_not_display_static_target(monkeypatch):
    from IPython import display

    from torch_world.envs import Maze

    shown = []
    monkeypatch.setattr(display, "display", shown.append)
    world = Maze(quality="low")

    def unavailable():
        raise ValueError("No verified reference")

    monkeypatch.setattr(world, "reference_policy", unavailable)
    with pytest.raises(ValueError, match="No verified reference"):
        world.show_success()
    assert not shown
    world.close()


def test_failed_success_policy_restores_limit_and_saves_nothing(tmp_path, monkeypatch):
    from IPython import display

    from torch_world.envs import Maze

    shown = []
    monkeypatch.setattr(display, "display", shown.append)
    world = Maze(quality="low", max_episode_steps=2)
    path = tmp_path / "failed.gif"
    with pytest.raises(ValueError, match="did not produce a verified success"):
        world.show_success(policy=lambda obs: torch.zeros(len(obs), 2), steps=4, path=path)
    assert world.max_episode_steps == 2
    assert not shown and not path.exists()
    world.close()
