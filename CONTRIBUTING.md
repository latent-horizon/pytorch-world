# Contributing

Thanks for helping build pytorch-world. Bug reports, fixes, new models, environments, datasets and documentation are all welcome.

## Development setup

```bash
git clone <repository-url> pytorch-world
cd pytorch-world
uv sync --extra examples --extra action --extra benchmark --extra newton --extra test
```

| extra | adds |
|---|---|
| `examples` | matplotlib, for the notebooks and `torch_world.plots` |
| `action` | MuJoCo, for the `RobotCube` arms |
| `test` | pytest |

## Checks

Run these before opening a pull request:

```bash
uvx ruff@0.17.0 format .
uvx ruff@0.17.0 check .
uv run pytest
```

Code is formatted and linted with [ruff](https://docs.astral.sh/ruff/) at a line length of 120.

## Adding a model

1. Subclass `JEPA`, `WorldModel` or `WorldActionModel` in a new file in its family's folder, for example `torch_world/models/jepa/_mymodel.py`. Build it from the shared modules in `nn`, `masks` and `losses`. See [Build your own model](docs/guides/custom-model.md).
2. Write the docstring: what the model predicts, its loss, and the paper and reference code. It becomes the model's API reference.
3. Export it from the family's `__init__.py` and from `torch_world.models`.
4. Add tests under `tests/models/`: output shapes, a finite loss, gradients and, for world models, a planning call.
5. Add a notebook under `examples/` that trains it on one of the [environments](docs/guides/environments.md) and shows what it learned.
6. Add a page in its family's folder under `docs/models/` (for example `docs/models/jepa/`), link it from the family page and add it to the sidebar in `docs/.vitepress/config.mts`.

A new family gets its own folder, `torch_world/models/<family>/` and `docs/models/<family>/`, with an index page that says what the family has in common.

## Adding environments and tasks

Use `EnvSpec` and `register_env` for lazy discovery; use `register_task` for a configuration variant. New dynamics can subclass `BenchmarkEnv`, which supplies collection, action validation, and cleanup. Keep optional simulator imports inside their adapters.

See the [benchmark environment guide](docs/guides/benchmark-environments.md) for the exact contract, a custom maze example, backend setup, and the shared experiment runner. Add seeded dynamics and terminal-boundary tests, a task description/reference, and a notebook using the library's interchangeable model and planner factories. Save measured runs through `run_experiment(output_dir=...)`; do not put invented results in unexecuted notebooks.

## Documentation

The site is built with [VitePress](https://vitepress.dev/) from `docs/`. API pages are generated from docstrings and the notebooks are rendered with their saved outputs, so run a notebook before committing it. `docs/.vitepress/prepare.py` does both before every build; it needs Node.js 20 or newer and `uv`.

```bash
npm install
npm run docs:dev
```

The site is also in Português, Español, 简体中文 and 日本語. Translations live in `docs/<locale>/` (`pt`, `es`, `zh`, `ja`) and mirror the English paths; a page without one shows the English text with a note. Interface strings are in `docs/.vitepress/i18n.ts`.

Pushes to `main` deploy the site to GitHub Pages. This file and [`docs/contributing.md`](docs/contributing.md) say the same thing; update both.

## Pull requests

- Keep each pull request to one change: a fix, a model, a feature.
- Add or update tests for the code you touch, and docs for anything user-facing.
- For a model, say how you checked it against the paper or the reference code.

By contributing, you agree that your contributions are licensed under the [Apache 2.0 license](LICENSE.md).
