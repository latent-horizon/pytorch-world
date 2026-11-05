"""Run evaluation episodes in a worker process that is replaced when an episode stalls."""

import json
import multiprocessing as mp
import queue
import time
import traceback
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


def run_episodes(
    episode: Callable[[Any], dict],
    seeds: Iterable,
    *,
    timeout: float,
    path: str | Path | None = None,
) -> list[dict]:
    """Run ``episode(seed)`` for every seed in one worker process, with a time limit per episode.

    A stall inside native code (for example MuJoCo's EGL rendering on a headless GPU server) cannot be interrupted
    from Python, so the episodes run in a spawned worker. The worker is reused from one episode to the next, keeping
    whatever ``episode`` caches (a loaded model, an open environment); when an episode exceeds ``timeout`` seconds the
    worker is killed, the episode is recorded with ``status="timeout"`` and a new worker carries on with the next seed.
    A worker that dies (a segfault) is replaced the same way, with ``status="crashed"``. An exception is recorded with
    ``status="error"``; a finished episode with ``status="ok"`` and the keys of its
    result. ``episode`` must be picklable, a function defined at the top level of a module.

    With ``path``, each record is appended to that JSON Lines file as soon as its episode ends, and seeds already
    recorded there are skipped, so an interrupted evaluation resumes where it stopped. Returns the records of all
    ``seeds``, in their order, including those read back from ``path``.
    """
    seeds = list(seeds)
    path = None if path is None else Path(path)
    done = {}
    if path is not None and path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                done[json.dumps(record["seed"])] = record
    context = mp.get_context("spawn")
    worker = None
    try:
        for seed in seeds:
            key = json.dumps(seed)
            if key in done:
                continue
            if worker is None:
                worker = _Worker(context, episode)
            tic = time.monotonic()
            worker.tasks.put(seed)
            status, value = worker.wait(tic + timeout)
            if status in ("timeout", "crashed"):
                worker.kill()
                worker = None
            record = {"seed": seed, "status": status, "seconds": round(time.monotonic() - tic, 3)}
            if status == "ok":
                record.update(value)
            elif value is not None:
                record["error"] = value
            done[key] = record
            if path is not None:
                with path.open("a") as file:
                    file.write(json.dumps(record) + "\n")
    finally:
        if worker is not None:
            worker.stop()
    return [done[json.dumps(seed)] for seed in seeds]


class _Worker:
    def __init__(self, context, episode):
        self.tasks, self.results = context.Queue(), context.Queue()
        self.process = context.Process(target=_serve, args=(episode, self.tasks, self.results), daemon=True)
        self.process.start()

    def wait(self, deadline: float) -> tuple[str, Any]:
        """The next result, or ``("timeout", None)`` after ``deadline``, or ``("crashed", ...)`` if the worker died."""
        while (left := deadline - time.monotonic()) > 0:
            try:
                return self.results.get(timeout=min(left, 1.0))
            except queue.Empty:
                if not self.process.is_alive():
                    return "crashed", f"worker exited with code {self.process.exitcode}"
        return "timeout", None

    def stop(self):
        self.tasks.put(None)
        self.process.join(5)
        self.kill()

    def kill(self):
        if self.process.is_alive():
            self.process.kill()
        self.process.join()


def _serve(episode, tasks, results):
    while (seed := tasks.get()) is not None:
        try:
            results.put(("ok", dict(episode(seed))))
        except Exception:  # noqa: BLE001 - any failure of one episode is recorded, and the next one runs
            results.put(("error", traceback.format_exc(limit=5)))
