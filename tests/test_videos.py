import pytest
import torch

from torch_world.datasets import random_clips, read_video


def test_random_clips_are_fixed_windows_over_time():
    frames = torch.arange(10.0)[:, None, None, None].expand(10, 3, 12, 16) + torch.arange(16.0) / 100
    clips = random_clips(frames, 5, length=4, size=6, stride=2, generator=torch.Generator().manual_seed(0))
    assert clips.shape == (5, 3, 4, 6, 6)
    time = clips[:, 0, :, 0, 0].floor()
    assert torch.equal(time[:, 1:] - time[:, :-1], torch.full((5, 3), 2.0))
    assert torch.allclose(clips[:, :, 0], clips[:, :, 1] - 2)  # the same window in every frame


def test_read_video(tmp_path):
    av = pytest.importorskip("av")
    path = tmp_path / "clip.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 32, 16, "yuv420p"
        for _ in range(3):
            container.mux(stream.encode(av.VideoFrame.from_ndarray(torch.zeros(16, 32, 3, dtype=torch.uint8).numpy())))
        container.mux(stream.encode())
    frames = read_video(path, size=(8, 16))
    assert frames.shape == (3, 3, 8, 16) and frames.max() <= 1
