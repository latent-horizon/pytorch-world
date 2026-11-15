"""Live training monitors: a dependency-free protocol with TensorBoard and Rerun backends.

A monitor has three methods: ``scalar(step, name, value)`` for a point on a curve, ``video(step, name, tensor)`` for
a clip ``(B, T, 3, H, W)`` in ``[0, 1]`` and ``close()`` at the end. ``NullMonitor`` is the default; the backends are
optional extras, ``pip install 'torch-world[tensorboard]'`` or ``pip install 'torch-world[rerun]'``, and constructing
one without its extra raises an ``ImportError`` with that hint. ``fit`` and the ``dreamzero`` train command accept
any monitor, and so does any loop of your own.
"""

import importlib.util
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor


class Monitor(Protocol):
    """A training monitor: ``scalar`` for a curve, ``video`` for a clip ``(B, T, 3, H, W)`` in ``[0, 1]``."""

    def scalar(self, step: int, name: str, value: float) -> None: ...

    def video(self, step: int, name: str, tensor: Tensor, fps: int = 12) -> None: ...

    def close(self) -> None: ...


class NullMonitor:
    """The default monitor: every method is a no-op, so monitoring stays opt-in."""

    def scalar(self, step: int, name: str, value: float) -> None:
        pass

    def video(self, step: int, name: str, tensor: Tensor, fps: int = 12) -> None:
        pass

    def close(self) -> None:
        pass


class TensorBoardMonitor:
    """Curves to TensorBoard event files under ``dir``; watch them with ``tensorboard --logdir dir``.

    Videos need ``moviepy``; without it, every frame becomes an image summary ``name/frame_000``, ... instead.
    """

    def __init__(self, dir: str | Path, **kwargs):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            raise ImportError(
                "TensorBoardMonitor needs the tensorboard extra: pip install 'torch-world[tensorboard]'"
            ) from e
        self._writer = SummaryWriter(str(dir), **kwargs)
        self._moviepy = importlib.util.find_spec("moviepy") is not None
        self._warned = False

    def scalar(self, step: int, name: str, value: float) -> None:
        self._writer.add_scalar(name, value, step)

    def video(self, step: int, name: str, tensor: Tensor, fps: int = 12) -> None:
        x = tensor.clamp(0, 1)
        if self._moviepy:
            self._writer.add_video(name, x, step, fps=fps)
            return
        if not self._warned:
            print("TensorBoardMonitor: moviepy is not installed; logging videos as one image per frame", flush=True)
            self._warned = True
        for t, frame in enumerate(x[0]):  # the first clip of the batch, one image summary per frame
            u8 = (frame * 255).clamp(0, 255).to(torch.uint8)
            self._writer.add_images(f"{name}/frame_{t:03d}", u8[None], step)

    def close(self) -> None:
        self._writer.close()


class RerunMonitor:
    """Curves and videos to a Rerun viewer; needs the rerun extra: ``pip install 'torch-world[rerun]'``.

    ``destination`` is ``"spawn"`` for a viewer on this machine, ``"rerun+http://host:9876/proxy"`` to stream to a
    remote one (run ``rerun`` on your laptop, train on the GPU box), or a ``.rrd`` path to record to file and open
    later with ``rerun file.rrd``. Scalars land on the ``step`` timeline and each ``video`` call appends its frames
    to the ``frame`` timeline, so one entity plays everything logged under that name, in order.
    """

    def __init__(self, destination: str | Path = "spawn"):
        try:
            import rerun as rr
        except ImportError as e:
            raise ImportError("RerunMonitor needs the rerun extra: pip install 'torch-world[rerun]'") from e
        self._rr = rr
        where = str(destination)
        rr.init("torch-world")
        if where == "spawn":
            rr.spawn()
        elif where.startswith(("rerun://", "rerun+http://", "rerun+https://")):
            rr.connect_grpc(where)
        else:
            Path(where).parent.mkdir(parents=True, exist_ok=True)
            rr.save(where)
        self._frame = 0

    def scalar(self, step: int, name: str, value: float) -> None:
        self._rr.set_time("step", sequence=step)
        self._rr.log(name, self._rr.Scalars([float(value)]))

    def video(self, step: int, name: str, tensor: Tensor, fps: int = 12) -> None:
        for b, clip in enumerate(tensor.detach().cpu().clamp(0, 1)):
            for t, frame in enumerate(clip):
                self._rr.set_time("step", sequence=step)
                self._rr.set_time("frame", duration=(self._frame + t) / fps)
                u8 = (frame * 255).round().clamp(0, 255).to(torch.uint8)
                self._rr.log(f"{name}/{b}", self._rr.Tensor(u8, dim_names=["channel", "height", "width"]))
        self._frame += tensor.size(1)

    def close(self) -> None:
        self._rr.disconnect()


def open_monitor(backend: str, dir: str | Path = "runs", rerun: str | Path = "spawn") -> Monitor:
    """A monitor by backend name: ``"none"``, ``"tensorboard"`` (event files under ``dir``) or ``"rerun"``.

    This is what the ``dreamzero`` train command's ``--monitor`` flag uses.
    """
    if backend == "none":
        return NullMonitor()
    if backend == "tensorboard":
        return TensorBoardMonitor(dir)
    if backend == "rerun":
        return RerunMonitor(rerun)
    raise ValueError(f"Unknown monitor backend {backend!r}; use 'none', 'tensorboard' or 'rerun'")
