import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("mujoco")

from torch_world.planning import CEM
from torch_world.sim2real import (
    ChessTaskConfig,
    SO101Config,
    SO101Robot,
    WorldModelPolicy,
    label_trial,
    load_checkpoint,
    make_adapter,
    run_experiment,
    run_trial,
)


def tiny_planner():
    return CEM(2, 4, num_samples=8, num_elites=2, iterations=1, bounds=(-1, 1))


def test_video_accepts_native_byte_and_normalized_frames(tmp_path):
    from PIL import Image

    from torch_world.sim2real.experiment import save_video

    frame = torch.tensor([32, 128, 224], dtype=torch.uint8)[:, None, None].expand(3, 8, 8)
    for name, pixels in (("byte", frame), ("float", frame.float() / 255)):
        path = tmp_path / f"{name}.gif"
        save_video(path, [pixels])
        with Image.open(path) as image:
            assert image.convert("RGB").getpixel((0, 0)) == (32, 128, 224)


def test_simulation_experiment_artifacts(tmp_path):
    config = SO101Config(size=16)
    result = run_experiment(
        config=config,
        smoke=True,
        episodes=1,
        updates=1,
        collect_steps=8,
        evaluation_steps=2,
        evaluation_seeds=(10007,),
        output_dir=tmp_path,
        save_gifs=False,
        planner_factory=tiny_planner,
    )
    path = Path(result["run_dir"])
    assert json.loads((path / "manifest.json").read_text())["complete"]
    recorded = json.loads((path / "results.json").read_text())
    assert recorded["physical_results"] is None
    assert recorded["training_episodes"][0]["seed"] != recorded["holdout_episodes"][0]["seed"]
    assert set(recorded["evaluation"]) == {"nominal", "randomized"}
    episodes = torch.load(path / "episodes.pt", weights_only=True)
    for episode in episodes["train"] + episodes["holdout"]:
        assert len(episode["pixels"]) == len(episode["actions"]) + 1
        assert episode["pixels"].dtype == torch.uint8
        assert episode["metadata"]["timestamps"] == sorted(episode["metadata"]["timestamps"])
    load_checkpoint(path / "model.pt", config, ChessTaskConfig())
    from torch_world.sim2real import RandomizationConfig
    from torch_world.sim2real.evaluation import evaluate_saved_run, restore_run

    restored, saved_config, saved_task, saved_calibration = restore_run(path)
    assert saved_config == config and saved_task == ChessTaskConfig() and saved_calibration is None
    assert restored.recipe["smoke"]
    rerun = evaluate_saved_run(
        path,
        randomization=RandomizationConfig(),
        output_dir=tmp_path / "reevaluation",
        seeds=(10008,),
        steps=1,
        planner_factory=tiny_planner,
    )
    assert rerun["training_run"] == str(path.resolve())
    assert Path(rerun["run_dir"], "manifest.json").exists()
    completed = [json.loads(line) for line in Path(rerun["run_dir"], "episodes.jsonl").read_text().splitlines()]
    assert len(completed) == 6
    assert {e["domain"] for e in completed} == {"nominal", "randomized"}
    assert all(e["seed"] == 10008 and "source" in e and "target" in e for e in completed)
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_saved_run(
            path,
            randomization=RandomizationConfig(),
            output_dir=tmp_path / "bad",
            seeds=(config.seed,),
            steps=1,
            planner_factory=tiny_planner,
        )


def test_trial_failure_preserves_artifacts_and_holds(calibration, tmp_path):
    config = SO101Config(size=16)
    with SO101Robot(config, ChessTaskConfig(), calibration) as robot:
        policy = WorldModelPolicy(make_adapter(size=16, smoke=True), robot.observe(), planner=tiny_planner())

        # A failed preflight must produce no trial motion and retain diagnostics.
        def fail(*args, **kwargs):
            raise TimeoutError("slow planner")

        policy.check_latency = fail
        with pytest.raises(TimeoutError, match="slow planner"):
            run_trial(robot, policy, output_dir=tmp_path, save_gif=False)
        assert robot.stopped and not robot.driver.sent
    folder = next(tmp_path.iterdir())
    trial = json.loads((folder / "trial.json").read_text())
    assert trial["success"] is None and "slow planner" in trial["failure"]
    with pytest.raises(ValueError, match="mock/dry"):
        label_trial(folder, True)


def test_retraining_preserves_splits_and_uses_only_training_data(tmp_path, monkeypatch):
    from torch_world.sim2real import retrain
    from torch_world.sim2real.experiment import save_checkpoint

    source = tmp_path / "source"
    source.mkdir()
    config, task = SO101Config(size=16), ChessTaskConfig()
    save_checkpoint(source / "model.pt", make_adapter(size=16, smoke=True), config, task)

    def episode(seed):
        return {
            "pixels": torch.randint(256, (6, 3, 16, 16), dtype=torch.uint8),
            "proprio": torch.zeros(6, 6),
            "actions": torch.zeros(5, 4),
            "requested_actions": torch.zeros(5, 4),
            "metadata": {"seed": seed},
        }

    torch.save({"train": [episode(7)], "holdout": [episode(1007)]}, source / "episodes.pt")
    original_data = (source / "episodes.pt").read_bytes()
    original_train = retrain.train

    def checked_train(adapter, episodes, **kwargs):
        assert [e.metadata["seed"] for e in episodes] == [7]
        return original_train(adapter, episodes, **kwargs)

    monkeypatch.setattr(retrain, "train", checked_train)
    path = retrain.retrain_saved_dataset(
        source, model="lewm", updates=1, batch_size=2, device="cpu", output_dir=tmp_path / "new"
    )
    assert (source / "episodes.pt").read_bytes() == (path / "episodes.pt").read_bytes() == original_data
    adapter = load_checkpoint(path / "model.pt", config, task)
    assert adapter.kind == "lewm" and not adapter.recipe["smoke"]
    assert len(json.loads((path / "losses.json").read_text())) == 1
    metrics = json.loads((path / "training-metrics.json").read_text())
    assert len(metrics) == 1 and metrics[0]["update"] == 1 and "pred_loss" in metrics[0]


def test_evaluate_selected_policy_preserves_task_assignment():
    from torch_world.sim2real import RandomizationConfig, evaluate

    adapter = make_adapter("lewam", size=16, smoke=True, chunk_size=2)
    config, task = SO101Config(size=16), ChessTaskConfig()
    progress = []
    metrics, frames = evaluate(
        adapter,
        config,
        task,
        RandomizationConfig(enabled=False),
        seeds=(10007,),
        steps=1,
        policies=("learned",),
        progress=progress.append,
    )
    assert set(metrics) == set(frames) == {"learned"}
    assert len(progress) == 1 and progress[0]["piece"] == "pawn" and progress[0]["color"] == "white"
    assert progress[0]["seed"] == 10007
    with pytest.raises(ValueError, match="policies"):
        evaluate(adapter, config, task, RandomizationConfig(), policies=())
