# Tutorial notebooks

Four notebooks teach catsim by doing. Each is a step up from the one before and is written for one reader; the [Tutorials page](https://douglasrizzo.com.br/catsim/tutorials.html) of the documentation says what each one teaches.

- [01_getting_started.ipynb](01_getting_started.ipynb) ([Colab](https://colab.research.google.com/github/douglasrizzo/catsim/blob/main/notebooks/01_getting_started.ipynb)): researchers new to catsim build an item bank, run a first simulation, read its result and follow one examinee's test.
- [02_comparing_strategies.ipynb](02_comparing_strategies.ipynb) ([Colab](https://colab.research.google.com/github/douglasrizzo/catsim/blob/main/notebooks/02_comparing_strategies.ipynb)): researchers compare estimators, selectors, exposure control and stoppers over replications.
- [03_live_sessions.ipynb](03_live_sessions.ipynb) ([Colab](https://colab.research.google.com/github/douglasrizzo/catsim/blob/main/notebooks/03_live_sessions.ipynb)): application developers run a test one response at a time, save and resume a session, and share item exposure across sessions.
- [04_custom_strategies.ipynb](04_custom_strategies.ipynb) ([Colab](https://colab.research.google.com/github/douglasrizzo/catsim/blob/main/notebooks/04_custom_strategies.ipynb)): method authors write a strategy of their own.

## Running them locally

From a clone of the repository:

```bash
uv sync --all-groups
uv run --with jupyterlab jupyter lab notebooks/
```

The dev group installs the Jupyter kernel and matplotlib, but no notebook front-end: `--with jupyterlab` adds JupyterLab for that command only. An editor that runs notebooks, such as VS Code, can use the project's environment, `.venv`, as the kernel instead.

Outside the repository, install catsim with its plot extra, which the figures need: `pip install 'catsim[plot]'`.

To change a notebook, see [CONTRIBUTING.md](../CONTRIBUTING.md): notebooks are committed without outputs, and the checks run them.
