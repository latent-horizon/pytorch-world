# Governance

pytorch-world is an open-source project supported by [Latent Horizon](https://github.com/latent-horizon). This document says who decides what, and how to take part in deciding.

## Roles

**Users** install and use the library. Questions, bug reports and feature requests in [issues](https://github.com/latent-horizon/pytorch-world/issues) are the most useful input the project gets.

**Contributors** have had a pull request merged: code, docs, a notebook, a benchmark result or a review. Anyone can become one; see [CONTRIBUTING.md](CONTRIBUTING.md).

**Maintainers** review and merge pull requests, triage issues, cut releases and keep the [roadmap](https://github.com/latent-horizon/pytorch-world/issues). They have write access to the repository and are listed below and in [`.github/CODEOWNERS`](.github/CODEOWNERS).

**The lead maintainer** has the final word when maintainers cannot agree, and owns the release, PyPI and documentation credentials.

| maintainer | role | areas |
|---|---|---|
| Kleyton da Costa ([@kleyt0n](https://github.com/kleyt0n)) | lead maintainer | all |

## Decisions

Most decisions are made in pull requests, by lazy consensus: a change is merged once a maintainer approves it, CI passes and no maintainer objects.

Some changes need an issue first, open for at least five working days, so that users and contributors can weigh in:

- a breaking change to a public API (anything exported from `torch_world` and its subpackages);
- a new required dependency, or a new optional extra;
- a new model family, environment suite or benchmark protocol;
- a change to this document, to the license or to the release process.

If maintainers disagree after discussion, they vote, one vote each, by simple majority; the lead maintainer breaks ties.

## What goes into the library

A model is added when it has a public paper, comes with tests and a notebook that trains it, and says how it was checked against the paper or the reference code. Results reported in the docs come from executed notebooks or from `run_experiment` outputs, never from unexecuted code. See [CONTRIBUTING.md](CONTRIBUTING.md#adding-a-model).

## Becoming a maintainer

A contributor with sustained, high-quality contributions (code, reviews or triage over several months) can be nominated by any maintainer. The nomination is accepted when no maintainer objects within one week. Maintainers who are inactive for six months move to emeritus status and can return by asking.

The project aims to have at least two maintainers from different organizations.

## Releases

Releases follow [semantic versioning](https://semver.org); before 1.0, a minor version (0.x) may change public APIs, and its release notes say how to migrate. A minor release ships about once a month, or sooner when a model lands; patch releases ship as soon as a fix is merged. Only the latest release gets fixes (see [SECURITY.md](SECURITY.md)). The [changelog](https://latent-horizon.github.io/pytorch-world/changelog) collects the release notes.

Any maintainer can cut a release:

1. bump `__version__` in `torch_world/__init__.py` on `main`, through a pull request;
2. tag that commit `vX.Y.Z` (or `vX.Y.ZrcN` for a pre-release) and push the tag.

The [release workflow](.github/workflows/release.yml) then checks that the tag matches `__version__`, runs CI, builds the wheel and sdist, publishes them to [PyPI](https://pypi.org/project/torch-world/) by trusted publishing (no stored token, with attestations), creates the GitHub release with generated notes and rebuilds the docs.

## Code of conduct

Everyone in the project's spaces follows the [Code of Conduct](CODE_OF_CONDUCT.md). Maintainers enforce it; reports go to the contact listed there, and a maintainer involved in a report steps out of handling it.
