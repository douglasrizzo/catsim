# Tutorials

The tutorials are four Jupyter notebooks that teach catsim by doing. They form a ladder: each is a
step up from the one before and is written for one reader, researchers first. Each notebook is the
hands-on version of a guide, which stays the reference for its task. Open a notebook in Colab, or
run it from a clone of the repository as the
[notebooks' README](https://github.com/douglasrizzo/catsim/tree/main/notebooks) explains.

## Getting started

**For** researchers new to catsim. **Level:** beginner.

Build an item bank, generated or read from a CSV file of calibrated parameters, and plot what its
items measure. Run a simulation whose precision target ends most tests, read its metrics with their
Monte Carlo errors and its tables by examinee, by step and by decile, follow one examinee's test item
by item, and repeat the run from its seed. The reference is {doc}`running-simulations`.

[View on GitHub](https://github.com/douglasrizzo/catsim/blob/main/notebooks/01_getting_started.ipynb) ·
[Open in Colab](https://colab.research.google.com/github/douglasrizzo/catsim/blob/main/notebooks/01_getting_started.ipynb)

## Comparing strategies

**For** researchers who have run a first simulation. **Level:** intermediate.

Compare strategies on one bank and one population, every design on the same seed, by differences
paired examinee by examinee and judged against their Monte Carlo errors: four estimators, three
selectors, four kinds of exposure control and two variable-length stoppers. Check one run's Monte
Carlo error against replications, and run a design in parts on several processes. The references are
{doc}`running-simulations` and {doc}`reproducibility`.

[View on GitHub](https://github.com/douglasrizzo/catsim/blob/main/notebooks/02_comparing_strategies.ipynb) ·
[Open in Colab](https://colab.research.google.com/github/douglasrizzo/catsim/blob/main/notebooks/02_comparing_strategies.ipynb)

## A test step by step

**For** researchers who want to see inside one test, and integrators who use catsim's computations
in an application. **Level:** intermediate.

Run one test with the engine's three steps, `start`, `select` and `answer`, read the session they
return, see why it ended and what an ended session does, end a test whose selector runs out of
items, pass exposure rates to the selector, and check that a simulation runs the same steps. The
reference is {doc}`embedding-catsim`.

[View on GitHub](https://github.com/douglasrizzo/catsim/blob/main/notebooks/03_a_test_step_by_step.ipynb) ·
[Open in Colab](https://colab.research.google.com/github/douglasrizzo/catsim/blob/main/notebooks/03_a_test_step_by_step.ipynb)

## Custom strategies

**For** method authors. **Level:** advanced.

Write a selector, a stopper and an exposure control of your own by subclassing catsim's base
classes, call one directly on a session, run them in a simulation, and see why a strategy that runs
on several processes must live in a module. The reference is {doc}`extending`.

[View on GitHub](https://github.com/douglasrizzo/catsim/blob/main/notebooks/04_custom_strategies.ipynb) ·
[Open in Colab](https://colab.research.google.com/github/douglasrizzo/catsim/blob/main/notebooks/04_custom_strategies.ipynb)
