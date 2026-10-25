"""Matplotlib figures for world models; requires ``pip install torch-world[examples]``.

One visual system across figures: a fixed categorical order, one sequential hue for time, hairline recessive axes,
2px lines, ringed markers and ink-colored text. In notebooks, figures render at retina resolution.
"""

from contextlib import contextmanager
from itertools import pairwise

import torch
from torch import Tensor

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, GRID, INK, INK_2, INK_3 = "#fcfcfb", "#e6e5e1", "#0b0b0b", "#52514e", "#8a8984"
GOAL, BASELINE = SERIES[1], "#cfcdc6"
SUCCESS, FAILURE = "#1f8a4c", "#c8364b"  # status colors, always shown with a ✓ or ✗ label


def _plt():
    import matplotlib.pyplot as plt

    _retina()
    return plt


def _retina():
    try:
        from IPython import get_ipython
        from matplotlib_inline.backend_inline import set_matplotlib_formats
    except ImportError:
        return
    if get_ipython() is not None:
        set_matplotlib_formats("retina")


@contextmanager
def style():
    plt = _plt()
    from cycler import cycler

    with plt.rc_context(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 120,
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.titlecolor": INK,
            "axes.titlelocation": "left",
            "axes.labelcolor": INK_2,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_3,
            "ytick.color": INK_3,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "legend.labelcolor": INK_2,
            "axes.prop_cycle": cycler(color=SERIES),
        }
    ):
        yield plt


def _image(x: Tensor):
    x = x.detach().float() / (255 if x.dtype == torch.uint8 else 1)
    return x.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def images(
    x: Tensor, titles: list[str] | None = None, ncols: int = 8, size: float = 1.2, rows: list[str] | None = None
):
    """Grid of ``(N, 3, H, W)`` images (float in ``[0, 1]`` or uint8), with optional ``titles`` and ``rows`` labels."""
    with style() as plt:
        nrows = -(-len(x) // ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(size * ncols + 0.3 * bool(rows), size * nrows + 0.25 * bool(titles)),
            squeeze=False,
            layout="constrained",
        )
        for i, ax in enumerate(axes.flat):
            ax.axis("off")
            if i < len(x):
                ax.imshow(_image(x[i]))
                if titles:
                    ax.set_title(titles[i], loc="center", fontweight="normal", color=INK_2, fontsize=8)
        for ax, label in zip(axes[:, 0], rows or []):
            ax.text(-0.06, 0.5, label, transform=ax.transAxes, rotation=90, ha="right", va="center", color=INK_2)
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
    _plt().close(fig)
    return fig


def masks(
    image: Tensor,
    grid: tuple[int, int],
    layers: dict[str, Tensor],
    size: float = 2.0,
    color: str | None = None,
    title: str | None = None,
):
    """The image beside one panel per mask, tinting the patches each index set holds on ``grid``.

    Each mask takes the next categorical color, or ``color`` for all of them when they show one mask over time.
    """
    image = image.detach().float().cpu()
    with style() as plt:
        fig, axes = plt.subplots(
            1, len(layers) + 1, figsize=(size * (len(layers) + 1), size + 0.3), layout="constrained"
        )
        axes[0].imshow(_image(image))
        axes[0].set_title(title or "image", loc="center")
        for ax, c, (name, idx) in zip(axes[1:], [color] * len(layers) if color else SERIES, layers.items()):
            cells = torch.zeros(grid[0] * grid[1])
            cells[idx.flatten().cpu()] = 1
            alpha = torch.nn.functional.interpolate(cells.view(1, 1, *grid), size=image.shape[-2:], mode="nearest")[0]
            tint = torch.tensor(_rgb(c))[:, None, None]
            gray = image.mean(0, keepdim=True).expand(3, -1, -1) * 0.35 + 0.6
            ax.imshow(_image(gray * (1 - alpha) + (0.45 * image + 0.55 * tint) * alpha))
            ax.set_title(f"{name}  ({int(cells.sum())}/{cells.numel()})", loc="center")
        for ax in axes:
            ax.axis("off")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
    _plt().close(fig)
    return fig


def curves(
    series: dict, smooth: int = 25, logy: bool = True, ylabel: str = "loss", size=(5.2, 2.6), xlabel: str = "step"
):
    """Training curves: the raw values as a faint trace under a ``smooth``-point moving average, labeled at the end.

    Each series is a list of values, one per step, or a ``(steps, values)`` pair when it is logged on some steps only.
    """
    with style() as plt:
        fig, ax = plt.subplots(figsize=size, layout="constrained")
        for color, (name, values) in zip(SERIES, series.items()):
            x, values = values if isinstance(values, tuple) else (range(len(values)), values)
            x, v = torch.tensor(list(x), dtype=torch.float32), torch.tensor(values, dtype=torch.float32)
            k = max(1, min(smooth, len(v)))
            avg = torch.nn.functional.avg_pool1d(v[None, None], k, 1)[0, 0]
            steps = x[k - 1 :]
            ax.plot(x, v, color=color, lw=0.8, alpha=0.18)
            ax.plot(steps, avg, color=color, label=name)
            ax.plot(steps[-1], avg[-1], "o", color=color, ms=5, mec=SURFACE, mew=1.5)
            ax.annotate(
                f"{avg[-1]:.3g}",
                (steps[-1], avg[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                color=INK_2,
            )
        ax.set(xlabel=xlabel, ylabel=ylabel)
        if logy:
            ax.set_yscale("log")
        if len(series) > 1:
            ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncols=len(series), handlelength=1.5, borderaxespad=0.2)
        ax.margins(x=0.08)

    _plt().close(fig)
    return fig


def trajectories(
    paths: Tensor,
    goals: Tensor,
    radius: float = 0.1,
    ncols: int = 4,
    size: float = 1.9,
    background: Tensor | None = None,
    scale: float = 1.0,
    success: Tensor | None = None,
):
    """Small multiples of planar paths ``(N, T, 2)`` in the unit square toward ``goals`` ``(N, 2)``.

    The path darkens with time on one sequential hue; the start is an open ring and the end a filled diamond, so both
    stay visible when they coincide, and the goal is a star inside its success disc of ``radius``. Each title marks
    the episode a success (✓) or failure (✗) and reports the distance to the goal at the start and at the end.
    ``success`` ``(N,)`` overrides the default test, ending inside the disc. Titles and the legend show distances
    times ``scale``, for example meters per unit of the square.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("time", SEQUENTIAL[1:])
    with style() as plt:
        n = len(paths)
        nrows = -(-n // ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(size * ncols, size * nrows + 0.1), squeeze=False, layout="constrained"
        )
        for i, ax in enumerate(axes.flat):
            ax.set(xlim=(0, 1), ylim=(1, 0), aspect="equal", xticks=[], yticks=[])
            ax.grid(False)
            for side in ax.spines.values():
                side.set(visible=True, color=GRID)
            if i >= n:
                ax.axis("off")
                continue
            path, goal = paths[i].cpu(), goals[i].cpu()
            if background is not None:
                ax.imshow(_image(background[i]), extent=(0, 1, 1, 0), alpha=0.25)
            ax.add_patch(plt.Circle(goal.tolist(), radius, color=GOAL, alpha=0.12, lw=0))
            segments = torch.stack([path[:-1], path[1:]], dim=1).numpy()
            ax.add_collection(
                LineCollection(
                    segments, cmap=cmap, array=torch.linspace(0, 1, len(segments)).numpy(), lw=2, capstyle="round"
                )
            )
            ax.plot(*goal, "*", color=GOAL, ms=13, mec=SURFACE, mew=1.5, zorder=3)
            ax.plot(*path[0], "o", ms=10, mfc=SURFACE, mec=INK_2, mew=2, zorder=4)
            ax.plot(*path[-1], "D", ms=6.5, color=SEQUENTIAL[-1], mec=SURFACE, mew=1.2, zorder=5)
            d0, d1 = (path[0] - goal).norm(), (path[-1] - goal).norm()
            ok = bool(success[i]) if success is not None else bool(d1 <= radius)
            ax.set_title(f"{'✓' if ok else '✗'}  {scale * d0:.2f} → {scale * d1:.2f}", color=INK if ok else INK_2)
        key = [
            plt.Line2D([], [], ls="", marker="o", ms=9, mfc=SURFACE, mec=INK_2, mew=2, label="start"),
            plt.Line2D([], [], color=SEQUENTIAL[3], lw=2, label="path (darker = later)"),
            plt.Line2D([], [], ls="", marker="D", ms=6.5, color=SEQUENTIAL[-1], mec=SURFACE, mew=1.2, label="end"),
            plt.Line2D([], [], ls="", marker="*", color=GOAL, ms=12, mec=SURFACE, mew=1.5, label="goal"),
            plt.Line2D(
                [], [], ls="", marker="o", color=GOAL, alpha=0.2, ms=12, label=f"success radius {scale * radius:.2g}"
            ),
        ]
        fig.legend(
            handles=key, loc="outside lower center", ncols=len(key), fontsize=8, handletextpad=0.4, columnspacing=1.4
        )
        fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04)
    _plt().close(fig)
    return fig


def distances(runs: dict[str, Tensor], radius: float | None = None, size=(5.2, 2.6)):
    """Distance to goal per step ``(N, T)`` for each method: the median as a line over the interquartile band."""
    with style() as plt:
        fig, ax = plt.subplots(figsize=size, layout="constrained")
        for color, (name, d) in zip(SERIES, runs.items()):
            d = d.cpu().float()
            q = torch.quantile(d, torch.tensor([0.25, 0.5, 0.75]), dim=0)
            steps = torch.arange(d.size(1))
            ax.fill_between(steps, q[0], q[2], color=color, alpha=0.12, lw=0)
            ax.plot(steps, q[1], color=color, label=name)
            ax.plot(steps[-1], q[1, -1], "o", color=color, ms=5, mec=SURFACE, mew=1.5)
            ax.annotate(
                f"{q[1, -1]:.2f}",
                (steps[-1], q[1, -1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                color=INK_2,
            )
        if radius is not None:
            ax.axhline(radius, color=INK_3, lw=0.8)
            ax.annotate(
                "success radius", (0, radius), xytext=(2, 3), textcoords="offset points", color=INK_3, fontsize=8
            )
        ax.set(xlabel="step", ylabel="distance to goal", ylim=(0, None))
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncols=len(runs), handlelength=1.5, borderaxespad=0.2)
        ax.margins(x=0.08)

    _plt().close(fig)
    return fig


def bars(groups: dict[str, dict[str, float]], xlabel: str = "accuracy", baseline: str | None = None, size=None):
    """Horizontal grouped bars: one row per key of ``groups``, one bar per series, each labeled with its value.

    Series take the categorical colors in order; the ``baseline`` series is drawn in neutral gray. Values in
    ``[0, 1]`` are shown as percentages. Values with ``low`` and ``high`` (``metrics.Rate``, e.g. from
    ``GoalReaching.intervals()``) get a whisker for their confidence interval.
    """
    rows = list(groups)
    names = list(groups[rows[0]])
    colors = iter(SERIES)
    colors = {n: BASELINE if n == baseline else next(colors) for n in names}
    percent = all(0 <= float(v) <= 1 for g in groups.values() for v in g.values())
    height = 0.8 / len(names)
    with style() as plt:
        fig, ax = plt.subplots(figsize=size or (5.6, 0.32 * len(rows) * len(names) + 0.9), layout="constrained")
        for j, name in enumerate(names):
            for i, row in enumerate(rows):
                y = len(rows) - 1 - i + ((len(names) - 1) / 2 - j) * height
                v = groups[row][name]
                ax.barh(y, float(v), height=0.86 * height, color=colors[name], label=name if i == 0 else None)
                end = float(v)
                if hasattr(v, "low"):
                    ax.plot([v.low, v.high], [y, y], color=INK_2, lw=1, solid_capstyle="butt")
                    ax.plot([v.low, v.low], [y - 0.2 * height, y + 0.2 * height], color=INK_2, lw=1)
                    ax.plot([v.high, v.high], [y - 0.2 * height, y + 0.2 * height], color=INK_2, lw=1)
                    end = v.high
                text = f"{float(v):.0%}" if percent else f"{float(v):.3g}"
                ax.annotate(text, (end, y), xytext=(4, 0), textcoords="offset points", va="center", color=INK_2)
        ax.set(yticks=range(len(rows)), yticklabels=rows[::-1], xlabel=xlabel)
        ax.set_xlim(0, (1 if percent else ax.get_xlim()[1]) * 1.1)
        if percent:
            ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_axisbelow(True)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncols=len(names), handlelength=1, borderaxespad=0.2)
    _plt().close(fig)
    return fig


def answers(
    x: Tensor, questions: list[str], predictions: list[str], truths: list[str], ncols: int = 4, size: float = 2.1
):
    """Question-answer cards: each image under its question, with the predicted answer marked right or wrong."""
    with style() as plt:
        nrows = -(-len(x) // ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(size * ncols, (size + 0.75) * nrows), squeeze=False, layout="constrained"
        )
        for i, ax in enumerate(axes.flat):
            ax.axis("off")
            if i >= len(x):
                continue
            ax.imshow(_image(x[i]))
            ax.set_title(questions[i], loc="left", fontsize=8, color=INK_2)
            right = predictions[i] == truths[i]
            ax.text(0, -0.05, ("✓ " if right else "✗ ") + predictions[i], transform=ax.transAxes, va="top", color=INK)
            if not right:
                ax.text(0, -0.17, f"truth: {truths[i]}", transform=ax.transAxes, va="top", color=INK_3, fontsize=8)
        fig.get_layout_engine().set(w_pad=0.06, h_pad=0.1)
    _plt().close(fig)
    return fig


def probes(true: Tensor, pred: Tensor, names: list[str], size: float = 2.5):
    """Linear-probe scatter per target ``(N, K)``: prediction against truth on the identity line, with R² in the title."""
    true, pred = true.detach().float().cpu(), pred.detach().float().cpu()
    r2 = 1 - (pred - true).square().sum(0) / (true - true.mean(0)).square().sum(0)
    with style() as plt:
        fig, axes = plt.subplots(1, len(names), figsize=(size * len(names), size + 0.3), layout="constrained")
        for i, (ax, name) in enumerate(zip(axes, names)):
            lo, hi = true[:, i].min().item(), true[:, i].max().item()
            ax.plot([lo, hi], [lo, hi], color=INK_3, lw=0.8)
            ax.scatter(true[:, i], pred[:, i], s=4, color=SERIES[0], alpha=0.35, lw=0)
            ax.set(title=f"{name}   R² = {r2[i]:.3f}", xlabel="true", aspect="equal")
            ax.set_ylabel("probe" if i == 0 else None)
    _plt().close(fig)
    return fig


class Animation:
    """A GIF that displays inline in notebooks and saves with ``save(path)``."""

    def __init__(self, data: bytes):
        self.data = data

    def _repr_html_(self) -> str:
        import base64

        return f'<img src="data:image/gif;base64,{base64.b64encode(self.data).decode()}" style="max-width: 100%"/>'

    def save(self, path: str):
        with open(path, "wb") as f:
            f.write(self.data)


def episodes(
    frames: Tensor,
    success: Tensor | list[bool] | None = None,
    titles: list[str] | None = None,
    ncols: int = 4,
    size: float = 1.9,
    fps: int = 4,
    hold: int = 4,
    dpi: int = 100,
    frame_steps: list[int] | None = None,
) -> Animation:
    """Videos of episodes ``(N, T, 3, H, W)`` as an animated grid, one tile per episode.

    With ``success`` ``(N,)``, each tile is labeled ✓ success or ✗ failure and framed in the matching status color.
    ``titles`` add a line per tile; the last frame is held for ``hold`` extra frames.
    ``frame_steps`` preserves episode step labels when displaying subsampled frames.
    """
    import io
    import tempfile

    from matplotlib.animation import FuncAnimation, PillowWriter

    n, steps = frames.shape[:2]
    frame_steps = list(range(steps)) if frame_steps is None else list(frame_steps)
    if len(frame_steps) != steps or not steps or any(b <= a for a, b in pairwise(frame_steps)):
        raise ValueError("frame_steps must contain one strictly increasing step number per frame")
    with style() as plt:
        nrows = -(-n // ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(size * ncols, size * nrows + 0.45), squeeze=False, layout="constrained"
        )
        images = []
        for i, ax in enumerate(axes.flat):
            ax.set(xticks=[], yticks=[])
            ax.grid(False)
            if i >= n:
                ax.axis("off")
                continue
            images.append(ax.imshow(_image(frames[i, 0])))
            label = titles[i] if titles else ""
            if success is not None:
                ok = bool(success[i])
                label = ("✓ success" if ok else "✗ failure") + (f"  ·  {label}" if label else "")
                for side in ax.spines.values():
                    side.set(visible=True, color=SUCCESS if ok else FAILURE, linewidth=2)
            else:
                for side in ax.spines.values():
                    side.set(visible=True, color=GRID)
            ax.set_title(label, loc="left", fontsize=8, color=INK)
        clock = fig.suptitle(f"step {frame_steps[0]} / {frame_steps[-1]}", x=0.01, ha="left", fontsize=9, color=INK_2)
        fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04)

        def update(t):
            t = min(t, steps - 1)
            for i, im in enumerate(images):
                im.set_data(_image(frames[i, t]))
            clock.set_text(f"step {frame_steps[t]} / {frame_steps[-1]}")
            return [*images, clock]

        animation = FuncAnimation(fig, update, frames=steps + hold, interval=1000 / fps, blit=False)
        with tempfile.NamedTemporaryFile(suffix=".gif") as file:
            animation.save(file.name, writer=PillowWriter(fps=fps), dpi=dpi)
            data = io.FileIO(file.name).readall()
    _plt().close(fig)
    return Animation(data)


class Video(Animation):
    """An MP4 that plays inline in notebooks, looping, and saves with ``save(path)``."""

    def _repr_html_(self) -> str:
        import base64

        data = base64.b64encode(self.data).decode()
        return f'<video autoplay loop muted playsinline controls style="width: 100%" src="data:video/mp4;base64,{data}"></video>'


def video(*panels: Tensor, fps: int = 6) -> Video:
    """An MP4 of videos ``(T, 3, H, W)`` in ``[0, 1]``, side by side with a white gap; requires ``av``."""
    import io

    import av

    gap = torch.ones(len(panels[0]), 3, panels[0].size(-2), 8)
    strip = torch.cat([x for panel in panels for x in (panel.detach().float().cpu(), gap)][:-1], dim=-1)
    strip = strip[..., : strip.size(-2) // 2 * 2, : strip.size(-1) // 2 * 2]  # yuv420p needs even sides
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps, options={"crf": "20"})
        stream.width, stream.height, stream.pix_fmt = strip.size(-1), strip.size(-2), "yuv420p"
        for frame in strip.clamp(0, 1).mul(255).byte().permute(0, 2, 3, 1).numpy():
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        container.mux(stream.encode())
    return Video(buffer.getvalue())


def pca_colors(tokens: Tensor, size: tuple[int, int] | None = None) -> Tensor:
    """The three leading principal components of patch tokens ``(N, h, w, D)`` as images ``(N, 3, h, w)``.

    One PCA is fitted over all tokens, so a color means the same thing in every image; each component is scaled to
    ``[0, 1]`` between its 1st and 99th percentiles. ``size`` upsamples the images bilinearly.
    """
    flat = tokens.detach().float().flatten(0, -2)
    flat = flat - flat.mean(0)
    pcs = flat @ torch.pca_lowrank(flat, q=3, center=False)[2]
    lo, hi = pcs.quantile(0.01, dim=0), pcs.quantile(0.99, dim=0)
    maps = ((pcs - lo) / (hi - lo)).clamp(0, 1).view(*tokens.shape[:-1], 3).permute(0, 3, 1, 2)
    if size is not None:
        maps = torch.nn.functional.interpolate(maps, size=size, mode="bilinear", align_corners=False).clamp(0, 1)
    return maps


def scatter(
    panels: dict[str, dict[str, float]],
    y: dict,
    ylabel: str = "success rate",
    titles: dict[str, str] | None = None,
    logx: tuple[str, ...] = (),
    ncols: int = 3,
    size: float = 2.4,
):
    """Small multiples of one measure ``y`` ``{series: value}`` against others: one panel per key of ``panels``, each
    holding ``{series: x}``, with one ringed point per series in its categorical color and a shared legend.

    Values of ``y`` with ``low`` and ``high`` (``metrics.Rate``) get a vertical whisker for their confidence interval,
    and values in ``[0, 1]`` are shown as percentages. ``titles`` adds a second line under a panel's name, for example
    its rank correlation with ``y``; panels named in ``logx`` use a log x axis.
    """
    names = list(y)
    colors = {name: SERIES[i % len(SERIES)] for i, name in enumerate(names)}
    percent = all(0 <= float(v) <= 1 for v in y.values())
    with style() as plt:
        nrows = -(-len(panels) // ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(size * ncols, (size + 0.25) * nrows + 0.35), squeeze=False, layout="constrained"
        )
        for i, ax in enumerate(axes.flat):
            if i >= len(panels):
                ax.axis("off")
                continue
            panel, xs = list(panels.items())[i]
            for name in names:
                if name not in xs:
                    continue
                x, v = float(xs[name]), y[name]
                if hasattr(v, "low"):
                    ax.plot([x, x], [v.low, v.high], color=colors[name], lw=1, alpha=0.45, solid_capstyle="butt")
                ax.plot(x, float(v), "o", color=colors[name], ms=6.5, mec=SURFACE, mew=1.2, zorder=3)
            ax.set_title(panel + (f"\n{titles[panel]}" if titles and panel in titles else ""))
            if panel in logx:
                ax.set_xscale("log")
                ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
                ax.xaxis.set_minor_formatter(lambda v, _: "")
            if percent:
                ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
            ax.set_ylabel(ylabel if i % ncols == 0 else None)
            ax.margins(x=0.12, y=0.08)
        key = [
            plt.Line2D([], [], ls="", marker="o", ms=6.5, color=colors[name], mec=SURFACE, mew=1.2, label=name)
            for name in names
        ]
        fig.legend(handles=key, loc="outside lower center", ncols=min(len(names), 4), handletextpad=0.3)
        fig.get_layout_engine().set(w_pad=0.04, h_pad=0.06)
    _plt().close(fig)
    return fig


def _rgb(hex_color: str) -> tuple[float, float, float]:
    return tuple(int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
