# Tutorials

The tutorials are four Jupyter notebooks that teach catsim by doing. They form a ladder: each is a
step up from the one before and is written for one reader, researchers first. Each notebook is the
hands-on version of a guide, which stays the reference for its task. Open a notebook in Colab, or
run it from a clone of the repository as the
[notebooks' README](https://github.com/douglasrizzo/catsim/tree/main/notebooks) explains.

## Getting started

**For** researchers new to catsim. **Level:** beginner.

Build an item bank, generated or from your own calibrated parameters, and plot what its items
measure. Run a simulation whose precision target ends most tests, read its result, follow one
examinee's test item by item, and repeat the run from one seed. The reference is
{doc}`running-simulations`.

[View on GitHub](https://github.com/douglasrizzo/catsim/blob/main/notebooks/01_getting_started.ipynb) ·
[Open in Colab](https://colab.research.google.com/github/douglasrizzo/catsim/blob/main/notebooks/01_getting_started.ipynb)

## Comparing strategies

**For** researchers who have run a first simulation. **Level:** intermediate.

Compare strategies on one bank and one population over seeded replications, by their paired
differences: four estimators, three selectors, four kinds of exposure control and two
variable-length stoppers, with the metrics that tell them apart and which criterion ended each
test. The references are {doc}`running-simulations` and {doc}`reproducibility`.

[View on GitHub](https://github.com/douglasrizzo/catsim/blob/main/notebooks/02_comparing_strategies.ipynb) ·
[Open in Colab](https://colab.research.google.com/github/douglasrizzo/catsim/blob/main/notebooks/02_comparing_strategies.ipynb)

## Live sessions

**For** application developers. **Level:** intermediate.

Run a test one response at a time, as an application does: choose each item, record the response,
read why the session stopped, save the session as JSON and resume it, and share item exposure
across sessions. The reference is {doc}`live-sessions`.

[View on GitHub](https://github.com/douglasrizzo/catsim/blob/main/notebooks/03_live_sessions.ipynb) ·
[Open in Colab](https://colab.research.google.com/github/douglasrizzo/catsim/blob/main/notebooks/03_live_sessions.ipynb)

## Custom strategies

**For** method authors. **Level:** advanced.

Write a strategy of your own by subclassing one of catsim's base classes, reading the session from
its context and, for an estimator, returning an estimate. The reference is {doc}`extending`.

[View on GitHub](https://github.com/douglasrizzo/catsim/blob/main/notebooks/04_custom_strategies.ipynb) ·
[Open in Colab](https://colab.research.google.com/github/douglasrizzo/catsim/blob/main/notebooks/04_custom_strategies.ipynb)
