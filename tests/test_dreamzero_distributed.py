import json
import os
import socket
import subprocess
import sys

import pytest
import torch

from torch_world.benchmarks.dreamzero import build_model, clip_geometry, lr_at, main, reach_demos
from torch_world.datasets import LatentClips, write_latent_shard

TINY = ["--dim", "24", "--depth", "1", "--num-heads", "3", "--patch-size", "2", "--attention", "sdpa"]
CONFIG = {
    "latent_channels": 4,
    "latent_size": [4, 4],
    "action_dim": 2,
    "chunk_frames": 2,
    "chunk_actions": 4,
    "num_chunks": 2,
}


def shards(root, clips=24, per_shard=10):
    generator = torch.Generator().manual_seed(0)
    for i, start in enumerate(range(0, clips, per_shard)):
        n = min(per_shard, clips - start)
        write_latent_shard(
            root / f"shard-{i:05d}.pt",
            latents=torch.randn(n, 5, 4, 4, 4, generator=generator),
            goal=torch.randn(n, 4, 4, 4, generator=generator),
            actions=torch.rand(n, 8, 2, generator=generator) * 2 - 1,
        )
    (root / "config.json").write_text(json.dumps(CONFIG))


def test_latent_clips_index_across_shards(tmp_path):
    shards(tmp_path)
    data = LatentClips(tmp_path)
    assert len(data) == 24 and data.offsets == [0, 10, 20, 24]
    whole = torch.cat([torch.load(p, weights_only=True)["actions"] for p in data.paths])
    for i in (0, 9, 10, 23):
        torch.testing.assert_close(data[i]["actions"], whole[i])
    assert data[23]["latents"].shape == (5, 4, 4, 4)
    with pytest.raises(IndexError):
        data[24]


def test_geometry_and_schedule():
    # Wan: 2 latent frames are 8 video frames; 8 actions per chunk is one action per frame.
    assert clip_geometry(2, 8, 4, 4) == (33, 1)
    assert clip_geometry(2, 16, 4, 4) == (33, 2)
    with pytest.raises(ValueError):
        clip_geometry(2, 6, 4, 4)
    assert lr_at(0, 100, 10) == 0.1 and lr_at(10, 100, 10) == 1.0 and lr_at(100, 100, 10) < 1e-9


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def torchrun(*args, nproc=2):
    # Gloo on loopback: macOS cannot resolve its own hostname for gloo's default interface.
    loopback = "lo0" if sys.platform == "darwin" else "lo"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "GLOO_SOCKET_IFNAME": loopback}
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--rdzv-backend",
        "c10d",
        "--rdzv-endpoint",
        f"127.0.0.1:{free_port()}",
        "--local-addr",
        "127.0.0.1",
        "--nproc-per-node",
        str(nproc),
        "-m",
        "torch_world.benchmarks.dreamzero",
        *args,
    ]
    result = subprocess.run(command, check=False, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return result


@pytest.mark.parametrize("strategy", ["ddp", "fsdp"])
def test_multi_process_training_resumes(tmp_path, strategy):
    shards(tmp_path / "data")
    out = tmp_path / "train"
    common = [
        "train",
        "--data",
        str(tmp_path / "data"),
        "--output",
        str(out),
        "--strategy",
        strategy,
        "--batch-size",
        "2",
        "--accumulate",
        "2",
        "--workers",
        "0",
        "--log-every",
        "1",
        "--save-every",
        "2",
        "--warmup",
        "1",
        *TINY,
    ]
    torchrun(*common, "--steps", "2")
    first = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in first] == [1, 2] and all(torch.isfinite(torch.tensor(r["loss"])) for r in first)
    torchrun(*common, "--steps", "4")
    log = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == [1, 2, 3, 4]
    checkpoint = torch.load(out / "checkpoint.pt", weights_only=False)
    assert checkpoint["step"] == 4
    config = json.loads((out / "config.json").read_text())
    model = build_model(config, "cpu")
    ema = torch.load(out / "model.pt", weights_only=True)
    model.load_state_dict(ema)
    assert model.goal_conditioned and model.dim == 24
    # model.pt holds the weight average, which lags the trained weights.
    assert any(not torch.equal(ema[k], v) for k, v in checkpoint["model"].items())
    assert all(torch.equal(ema[k], checkpoint["ema"][k]) for k in ema)
    if strategy == "ddp":
        main(["diagnose", "--run", str(out), "--data", str(tmp_path / "data"), "--clips", "6", "--batch-size", "4"])
        report = json.loads((out / "diagnose.json").read_text())
        assert report["clips"] == 6 and all(v >= 0 for v in report.values())


def test_reach_demonstrations_move_toward_aims():
    from torch_world.envs import RobotCube

    env = RobotCube(batch_size=2, task="reach", size=32, control_dt=0.2, max_step=0.1)
    data = reach_demos(env, 6, torch.Generator().manual_seed(0), noise=0.05, retarget=3)
    assert data["pixels"].shape == (2, 7, 3, 32, 32) and data["actions"].shape == (2, 6, 4)
    assert (data["actions"][..., 3] == 1).all() and data["actions"].abs().max() <= 1
