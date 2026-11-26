# Tutorial notebooks

Four notebooks teach catsim by doing. Each is a step up from the one before and is written for one reader; the [Tutorials page](https://douglasrizzo.com.br/catsim/tutorials.html) of the documentation says what each one teaches.

- [01_getting_started.ipynb](01_getting_started.ipynb) ([Colab](https://colab.research.google.com/github/douglasrizzo/catsim/blob/main/notebooks/01_getting_started.ipynb)): researchers new to catsim build an item bank, generated or read from a CSV file, run a first simulation, read its metrics and tables, and follow one examinee's test.
- [02_comparing_strategies.ipynb](02_comparing_strategies.ipynb) ([Colab](https://colab.research.google.com/github/douglasrizzo/catsim/blob/main/notebooks/02_comparing_strategies.ipynb)): researchers compare estimators, selectors, exposure control and stoppers on the same seed, judging each difference against its Monte Carlo error, and run a design in replications and on several processes.
- [03_a_test_step_by_step.ipynb](03_a_test_step_by_step.ipynb) ([Colab](https://colab.research.google.com/github/douglasrizzo/catsim/blob/main/notebooks/03_a_test_step_by_step.ipynb)): researchers and integrators run one test with `start`, `select` and `answer`, read the session and why it ended, and pass exposure rates to the selector.
- [04_custom_strategies.ipynb](04_custom_strategies.ipynb) ([Colab](https://colab.research.google.com/github/douglasrizzo/catsim/blob/main/notebooks/04_custom_strategies.ipynb)): method authors write a selector, a stopper and an exposure control of their own, and see why a strategy that runs on several processes must live in a module.

## Running them locally

From a clone of the repository:

```bash
uv sync --all-groups
uv run --with jupyterlab jupyter lab notebooks/
```

The dev group installs the Jupyter kernel and matplotlib, but no notebook front-end: `--with jupyterlab` adds JupyterLab for that command only. An editor that runs notebooks, such as VS Code, can use the project's environment, `.venv`, as the kernel instead.

Outside the repository, install catsim with its plot extra, which the figures need: `pip install 'catsim[plot]'`.

To change a notebook, see [CONTRIBUTING.md](../CONTRIBUTING.md): notebooks are committed without outputs, and the checks run them.
