# Figures

`torch_world.plots` draws the figures used in the notebooks, in one visual system: a fixed categorical order, one sequential hue for time, hairline axes and 2px lines. It needs matplotlib:

```bash
pip install "torch-world[examples]"
```

Every function returns a matplotlib `Figure`, which notebooks display (at retina resolution) and scripts can save with `fig.savefig(...)`.

| function | draws |
|---|---|
| `images(x, titles, ncols)` | a grid of `(N, 3, H, W)` images |
| `masks(image, grid, layers)` | an image beside one panel per mask, tinting the kept patches |
| `curves(series, smooth)` | training curves: raw values under a moving average, labeled at the end |
| `trajectories(paths, goals)` | small multiples of planar paths toward goals, with a success radius |
| `distances(runs, radius)` | distance to goal per step: median over the interquartile band, per method |
| `bars(groups, baseline)` | horizontal grouped bars with value labels |
| `answers(x, questions, predictions, truths)` | question-answer cards marked right or wrong |
| `probes(true, pred, names)` | linear-probe scatter per target, with R² |

```python
from torch_world import plots

log = {"prediction": [], "SIGReg": []}
...  # training
plots.curves(log, smooth=100)
```

For custom figures in the same style, use the context manager and the palette:

```python
with plots.style() as plt:
    fig, ax = plt.subplots()
    ax.plot(values, color=plots.SERIES[0])
```
