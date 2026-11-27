# Contributing

**catsim** is built in an object-oriented paradigm (or as object-oriented as Python allows) so it is rather simple to extend it. You can write new [initializers](https://douglasrizzo.com.br/catsim/api/techniques/initialization/index.html), [selectors](https://douglasrizzo.com.br/catsim/api/techniques/selection/index.html), [estimators](https://douglasrizzo.com.br/catsim/api/techniques/estimation/index.html) or [stopping criteria](https://douglasrizzo.com.br/catsim/api/techniques/stopping/index.html) by extending the base classes that are present in each of the corresponding modules. You can also write new [IRT-related functions](https://douglasrizzo.com.br/catsim/api/irt.html), as long as you have the right academic papers to prove they are relevant.

If you think the [CAT engine](https://douglasrizzo.com.br/catsim/api/engine.html) or [`simulate`](https://douglasrizzo.com.br/catsim/api/simulation.html) could be doing something it is not doing, feel free to study it and make a contribution.

If you know a better way to present all the data collected during simulation, feel free to contribute with your own [plots](https://douglasrizzo.com.br/catsim/api/plot.html).

## Psychometrics

**catsim** still has a way to go before it can be considered a mature package. Here is a list of features that may be of interest:

- Other test evaluation methods;
- Comparisons between simulation results (for example [^Barr2010]);
- Other item selection criteria (see [^Lind2010] for a survey).

## Development setup

Clone the repository, then install the development tools with [uv](https://docs.astral.sh/uv/) and the pre-commit hooks:

```bash
uv sync --group dev
uv run pre-commit install
```

The first command installs catsim with its runtime dependencies, declared in `pyproject.toml`, and the tools for tests, linting, type checking, coverage and notebooks. pre-commit then runs the repository's formatters and checks on every commit.

## Quality gate

Changes that alter behavior come with new or updated tests in `tests/`. Before you send a pull request, run the tests and every pre-commit hook, as CI does:

```bash
uv run pytest -q
uv run pre-commit run --all-files
```

pytest also runs the docstring examples in `src/catsim`, and turns warnings into errors. The hooks run Ruff, ty, pylint, import-linter (the layer contracts in `pyproject.toml`), vulture, typos and the notebooks.

The tutorials are `.ipynb` files in `notebooks/`, versioned with the package so that they open from GitHub and Colab and are checked as the API changes. Commit them without outputs: pre-commit strips the outputs (nbstripout) and runs the notebooks a commit changes (nbmake), and CI runs every notebook, so a notebook that fails to run fails the gate like a failing test. If you change a notebook, run the notebooks too:

```bash
uv run pytest --nbmake --nbmake-timeout=300 notebooks/*.ipynb
```

## Building the documentation

The documentation sources are in `sphinx/`. Build the HTML pages with the command below, which also installs Sphinx and its extensions (the `docs` dependency group):

```bash
uv run --group docs sphinx-build -b html -W --keep-going sphinx sphinx/_build/html
```

`-W` turns warnings into errors.

External links are checked once a week by the "Link check" workflow (`.github/workflows/linkcheck.yml`), not in pull requests, and a broken link fails that run. To check them yourself, which takes several minutes because github.com rate-limits automated clients:

```bash
uv run --group docs sphinx-build -b linkcheck sphinx sphinx/_build/linkcheck
```

Links to the Python, NumPy, Matplotlib and SciPy documentation come from those projects' intersphinx inventories. The build downloads each inventory and, when a site cannot be reached, reads the copy stored in `sphinx/_inventories/`. Refresh the stored copies with:

```bash
cd sphinx && for p in python=https://docs.python.org/3/ numpy=https://numpy.org/doc/stable/ matplotlib=https://matplotlib.org/stable/ scipy=https://docs.scipy.org/doc/scipy/; do curl -fsSL -o "_inventories/${p%%=*}.inv" "${p#*=}objects.inv"; done
```

## Releasing

Bump the version with `uv version`: `pyproject.toml` is its only source. A release is ready only when its documentation is online, so a documentation deployment gates every publication, in the same workflow run as the publication (a job cannot wait for a job in another run): the `dev` → `main` pull request deploys the dev documentation and then publishes a release candidate to TestPyPI, and a GitHub release deploys the production documentation and then publishes to PyPI. These are the only two publishing paths; there is no manual publish. The production documentation goes live before the PyPI upload waits for the maintainer's approval in the `pypi` environment, so a rejected upload leaves documentation online for a version that is not on PyPI until the next deployment.

Each GitHub release has a hand-written summary (the highlights, the breaking changes with what users must do, and the results that change for the same seed), followed by GitHub's generated list of pull requests. The release notes are the only record of what changed: there is no CHANGELOG file.

## Adding a technique

Each implemented technique has a page under `sphinx/api/techniques/` with its definition, its parameters and its Guarantees section. A new technique starts as a technique card in the [backlog](https://github.com/douglasrizzo/catsim/tree/dev/backlog), written from the [card template](https://github.com/douglasrizzo/catsim/blob/dev/backlog/_template.md); the [backlog's README](https://github.com/douglasrizzo/catsim/blob/dev/backlog/README.md) explains how a technique card is researched, revalidated, implemented and then promoted into the documentation. The template's Definition and References sections set the citation convention: cite the page, equation, table or figure of each formula where the formula appears, cite only locations you have read, say where the implementation departs from its sources, and give every reference its DOI and an open copy when one exists. A card writes its references by hand; a technique page generates them from [`sphinx/references.bib`](https://github.com/douglasrizzo/catsim/blob/dev/sphinx/references.bib). On a technique page, cite with `` {cite:t}`key{p. N}` `` (or `{cite:p}` in parentheses), add each source to `references.bib` with its DOI and an `Open copy: \url{...}`, `Open access` or `Not openly available; ...` note, and end the page's References section with a `{bibliography}` filtered to the page (`:filter: docname in docnames`), as the [technique-page template](https://github.com/douglasrizzo/catsim/blob/dev/sphinx/api/techniques/_template.md) shows.

## Parity tests

`tests/test_parity_catr.py` checks catsim against the R package [catR](https://cran.r-project.org/package=catR) on fixed cases ([ADR-0020](sphinx/adr/0020-catr-as-the-reference.md)). It runs with the rest of the suite and needs no R: catR's outputs are JSON files in `tests/parity/catr/`, written with the R and catR versions by the R scripts beside them. The cases are at three levels:

1. single computations (`irt.R`, `estimation.R`, `selection.R`, `stopping.R`);
1. whole tests on recorded responses (`whole_tests.R`), which must give the same items in the same order and the same estimates;
1. simulations of the randomized selectors (`simulations.R`), whose metrics must agree within their Monte Carlo error, since R and NumPy draw different numbers.

Each case records catR's settings, and the test builds the catsim strategy that matches them. A deliberate departure that no setting reproduces is a strict expected failure, listed with its reason in the test's `_DEPARTURES`, and the technique page's catR dropdown states it: if catsim comes to agree with catR there, the run fails until the departure is removed from both. A new technique with a catR counterpart comes with its parity cases.

Regenerate the files when the cases change or the pinned catR version (`CATR_VERSION` in `generate.R`) moves, and review the diff of the JSON files:

```bash
Rscript tests/parity/catr/generate.R
```

The script needs R and takes about two minutes. It installs the pinned catR into `~/.cache/catsim-parity/R-lib`, outside the repository and your own R library; set `CATSIM_CATR_LIB` to use another directory. Every run writes the same files.

## How to contribute

**Contributing code:** create a fork on GitHub, make your changes on your own repo and then send a pull request to our **dev** branch so that we can check your contributions. Make sure your changes pass the quality gate above. Pull requests are squash-merged into **dev**, with the pull request's title and description as the commit message, so your branch's commits need no tidying.

**Contributing ideas:** file an [issue on GitHub](https://github.com/douglasrizzo/catsim/issues), label it as an **enhancement** and describe as thoroughly as possible what could be done.

**Blaming me:** file an [issue on GitHub](https://github.com/douglasrizzo/catsim/issues) describing as thoroughly as possible the problem, with error messages, descriptions of your tests and possibly suggestions for fixing it.

## References

[^Barr2010]: Barrada, J. R., Olea, J., Ponsoda, V., & Abad, F. J. (2010). A Method for the Comparison of Item Selection Rules in Computerized Adaptive Testing. Applied Psychological Measurement, 34(6), 438–452. <https://doi.org/10.1177/0146621610370152>

[^Lind2010]: Linden, W. J. V. D., & Pashley, J. P. Item Selection and Ability Estimation in Adaptive Testing. _In_ Linden, W. J. V. D., & Glas, C. A. W. (2010). Elements of Adaptive Testing. New York, NY, USA: Springer New York.
