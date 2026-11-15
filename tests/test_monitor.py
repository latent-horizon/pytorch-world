import subprocess
import sys

import pytest
import torch
from torch.utils.data import DataLoader

from torch_world.datasets import TrajectoryDataset
from torch_world.envs import TwoRooms
from torch_world.losses import SIGReg
from torch_world.models import LeWM
from torch_world.nn import CLSEncoder
from torch_world.utils import NullMonitor, fit, open_monitor


class Recording:
    """A monitor that records every scalar call, to check what the trainers send."""

    def __init__(self):
        self.scalars: list[tuple[int, str, float]] = []

    def scalar(self, step: int, name: str, value: float) -> None:
        self.scalars.append((step, name, value))

    def video(self, step: int, name: str, tensor: torch.Tensor, fps: int = 12) -> None:
        pass

    def close(self) -> None:
        pass


def lewm(action_dim=2):
    encoder = CLSEncoder(img_size=16, patch_size=8, embed_dim=32, depth=1, num_heads=2)
    return LeWM(
        encoder, action_dim, embed_dim=32, depth=1, num_heads=2, head_dim=16, mlp_dim=64, proj_dim=64, sigreg=SIGReg(32)
    )


def loader():
    data = TwoRooms(batch_size=16, size=16).collect(8, generator=torch.Generator().manual_seed(0))
    return DataLoader(TrajectoryDataset(data, 4), batch_size=32, shuffle=True, drop_last=True)


def test_fit_sends_every_loss_to_the_monitor():
    rec = Recording()
    train = loader()
    batches = len(train)
    log = fit(lewm(), train, epochs=2, monitor=rec)
    assert len(rec.scalars) == len(log) * 2 * batches
    assert {name for _, name, _ in rec.scalars} == set(log)
    steps = [step for step, _, _ in rec.scalars]
    assert sorted(set(steps)) == list(range(2 * batches)) and steps == sorted(steps)
    for step, name, value in rec.scalars:
        assert value == pytest.approx(log[name][step])


def test_null_monitor_and_open_monitor_none_are_no_ops():
    monitor = open_monitor("none")
    assert isinstance(monitor, NullMonitor)
    monitor.scalar(0, "loss", 1.0)
    monitor.video(0, "rollout", torch.rand(1, 4, 3, 16, 16))
    monitor.close()


def test_open_monitor_rejects_unknown_backends():
    with pytest.raises(ValueError, match="Unknown monitor backend"):
        open_monitor("wandb")


def test_tensorboard_monitor_writes_events(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from torch_world.utils import TensorBoardMonitor

    monitor = TensorBoardMonitor(tmp_path)
    for step in range(3):
        monitor.scalar(step, "train/loss", 1.0 / (step + 1))
    monitor.video(10, "rollout/true", torch.rand(2, 4, 3, 16, 16))
    monitor.close()
    acc = EventAccumulator(str(tmp_path))
    acc.Reload()
    points = acc.Scalars("train/loss")
    assert [p.step for p in points] == [0, 1, 2] and [p.value for p in points] == pytest.approx([1.0, 0.5, 1 / 3])
    images = acc.Tags()["images"]
    assert any(tag.startswith("rollout/true") for tag in images), images  # one per frame without moviepy, a GIF with


def test_rerun_monitor_records_scalars_and_video(tmp_path):
    pytest.importorskip("rerun")
    from torch_world.utils import RerunMonitor

    path = tmp_path / "monitor.rrd"
    monitor = RerunMonitor(path)
    for step in range(3):
        monitor.scalar(step, "train/loss", 1.0 / (step + 1))
    monitor.video(10, "rollout/true", torch.rand(2, 4, 3, 16, 16))
    monitor.video(20, "rollout/true", torch.rand(2, 4, 3, 16, 16))
    monitor.close()
    assert path.stat().st_size > 0
    stats = subprocess.run(
        [sys.executable, "-m", "rerun", "rrd", "stats", str(path)], capture_output=True, text=True, check=True
    )
    assert "UNSORTED" not in stats.stdout  # the frame timeline must stay monotonic across video calls
    entities = stats.stdout.split("Num chunks per index")[0]
    for entity in ("/train/loss", "/rollout/true/0", "/rollout/true/1"):
        assert entity in entities, entity
