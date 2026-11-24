# Contributing

**catsim** is built in an object-oriented paradigm (or as object-oriented as Python allows) so it is rather simple to extend it. You can write new [initializers](https://douglasrizzo.com.br/catsim/api/techniques/initialization/index.html), [selectors](https://douglasrizzo.com.br/catsim/api/techniques/selection/index.html), [estimators](https://douglasrizzo.com.br/catsim/api/techniques/estimation/index.html) or [stopping criteria](https://douglasrizzo.com.br/catsim/api/techniques/stopping/index.html) by extending the base classes that are present in each of the corresponding modules. You can also write new [IRT-related functions](https://douglasrizzo.com.br/catsim/api/irt.html), as long as you have the right academic papers to prove they are relevant.

If you think the [CAT engine](https://douglasrizzo.com.br/catsim/api/engine.html) or the [simulation runner](https://douglasrizzo.com.br/catsim/api/simulation.html) could be doing something it is currently not doing, feel free to study it and make a contribution.

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

pre-commit then runs the repository's formatters and checks on every commit.

## Quality gate

Changes that alter behavior come with new or updated tests in `tests/`. Before you send a pull request, run the checks listed in [AGENTS.md](https://github.com/douglasrizzo/catsim/blob/dev/AGENTS.md):

```bash
uv run pytest -q
uv run ruff check src/catsim tests
uv run pylint src/catsim
uv run ty check
```

The tutorials are `.ipynb` files in `notebooks/`, versioned with the package so that they open from GitHub and Colab and are checked as the API changes. Commit them without outputs: pre-commit strips the outputs (nbstripout) and runs the notebooks a commit changes (nbmake), and CI runs every notebook, so a notebook that no longer runs fails the gate like a failing test. If you change a notebook, run the notebooks too:

```bash
uv run pytest --nbmake --nbmake-timeout=300 notebooks/*.ipynb
```

## Building the documentation

The documentation sources are in `sphinx/`. Build the HTML pages with:

```bash
uv run --group docs sphinx-build -b html -W --keep-going sphinx sphinx/_build/html
```

`-W` turns warnings into errors. The build also regenerates `README.md` from `sphinx/readme_head.md` and `sphinx/readme_body.md`, so edit those two files instead of `README.md`.

Links to the Python, NumPy, Matplotlib and SciPy documentation come from those projects' intersphinx inventories. The build downloads each inventory and, when a site cannot be reached, reads the copy stored in `sphinx/_inventories/`. Refresh the stored copies with:

```bash
cd sphinx && for p in python=https://docs.python.org/3/ numpy=https://numpy.org/doc/stable/ matplotlib=https://matplotlib.org/stable/ scipy=https://docs.scipy.org/doc/scipy/; do curl -fsSL -o "_inventories/${p%%=*}.inv" "${p#*=}objects.inv"; done
```

A release is ready only when its documentation is online, so a documentation deployment gates every publication, in the same workflow run as the publication (a job cannot wait for a job in another run): the `dev` → `main` pull request deploys the dev documentation and then publishes a release candidate to TestPyPI, and a GitHub release deploys the production documentation and then publishes to PyPI. These are the only two publishing paths; there is no manual publish. The production documentation goes live before the PyPI upload waits for the maintainer's approval in the `pypi` environment, so a rejected upload leaves documentation online for a version that is not on PyPI until the next deployment.

## Adding a technique

Each implemented technique has a page under `sphinx/api/techniques/` with its definition, parameters and behavioral contracts. A new technique starts as a spec in [`specs/`](https://github.com/douglasrizzo/catsim/tree/dev/specs), written from the [spec template](https://github.com/douglasrizzo/catsim/blob/dev/specs/_template.md); [`specs/README.md`](https://github.com/douglasrizzo/catsim/blob/dev/specs/README.md) explains how a spec is revalidated, implemented and then promoted into the documentation. The template's Definition and References sections set the citation convention: cite the page, equation, table or figure of each formula where the formula appears, cite only locations you have read, say where the implementation departs from its sources, and give every reference its DOI and an open copy when one exists.

## How to contribute

**Contributing code:** create a fork on GitHub, make your changes on your own repo and then send a pull request to our **dev** branch so that we can check your contributions. Make sure your changes pass the quality gate above.

**Contributing ideas:** file an [issue on GitHub](https://github.com/douglasrizzo/catsim/issues), label it as an **enhancement** and describe as thoroughly as possible what could be done.

**Blaming me:** file an [issue on GitHub](https://github.com/douglasrizzo/catsim/issues) describing as thoroughly as possible the problem, with error messages, descriptions of your tests and possibly suggestions for fixing it.

## References

[^Barr2010]: Barrada, J. R., Olea, J., Ponsoda, V., & Abad, F. J. (2010). A Method for the Comparison of Item Selection Rules in Computerized Adaptive Testing. Applied Psychological Measurement, 34(6), 438–452. <http://doi.org/10.1177/0146621610370152>

[^Lind2010]: Linden, W. J. V. D., & Pashley, J. P. Item Selection and Ability Estimation in Adaptive Testing. _In_ Linden, W. J. V. D., & Glas, C. A. W. (2010). Elements of Adaptive Testing. New York, NY, USA: Springer New York.
