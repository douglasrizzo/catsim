<p align="center">
  <img src="sphinx/logo_text.svg?sanitize=true" alt="Logo" />
</p>

------------------------------------------------------------------------

[![Unit tests](https://github.com/douglasrizzo/catsim/actions/workflows/test-on-push.yml/badge.svg)](https://github.com/douglasrizzo/catsim/actions/workflows/test-on-push.yml)
[![Test Coverage](https://coveralls.io/repos/github/douglasrizzo/catsim/badge.svg?branch=master)](https://coveralls.io/github/douglasrizzo/catsim?branch=master)
[![Latest Version](https://badge.fury.io/py/catsim.svg)](https://badge.fury.io/py/catsim)
[![Requirements Status](https://requires.io/github/douglasrizzo/catsim/requirements.svg?branch=master)](https://requires.io/github/douglasrizzo/catsim/requirements/?branch=master)
[![Digital Object Identifier](https://zenodo.org/badge/doi/10.5281/zenodo.46420.svg)](http://dx.doi.org/10.5281/zenodo.46420)

**catsim** is a Python package for computerized adaptive testing (CAT)
simulations. It provides multiple methods for:

- [test initialization](https://douglasrizzo.github.io/catsim/initialization.html) (selecting the initial proficiency of the examinees)
- [item selection](https://douglasrizzo.github.io/catsim/selection.html)
- [proficiency estimation](https://douglasrizzo.github.io/catsim/estimation.html)
- [test stopping](https://douglasrizzo.github.io/catsim/stopping.html)

These methods can either be used in a standalone fashion
[\[1\]](https://douglasrizzo.github.io/catsim/introduction.html#autonomous-usage)
to power other software or be used with *catsim* to simulate the
application of computerized adaptive tests
[\[2\]](https://douglasrizzo.github.io/catsim/introduction.html#running-simulations),
given a sample of examinees, represented by their proficiency levels,
and an item bank, represented by their parameters according to some
[logistic Item Response Theory
model](https://douglasrizzo.github.io/catsim/introduction.html#item-response-theory-models).

## What's a CAT

Computerized adaptive tests are educational evaluations, usually taken
by examinees in a computer or some other digital means, in which the
examinee\'s proficiency is evaluated after the response of each item.
The new proficiency is then used to select a new item, closer to the
examinee\'s real proficiency. This method of test application has
several advantages compared to the traditional paper-and-pencil method
or even linear tests applied electronically, since high-proficiency
examinees are not required to answer all the easy items in a test,
answering only the items that actually give some information regarding
his or hers true knowledge of the subject at matter. A similar, but
inverse effect happens for those examinees of low proficiency level.

More information is available [in the
docs](https://douglasrizzo.github.io/catsim/introduction.html) and over
at
[Wikipedia](https://en.wikipedia.org/wiki/Computerized_adaptive_testing).

## Installation

Install it using `pip install catsim`.

## Basic Usage

**NEW:** there is now [a Colab Notebook](https://colab.research.google.com/drive/14zEWoDudBCXF0NO-qgzoQpWUGBcJ2lPH?usp=sharing) teaching the basics of catsim!

1.  Have an [item matrix](https://douglasrizzo.github.io/catsim/item_matrix.html);
2.  Have a sample of examinee proficiencies, or a number of examinees to be generated;
3.  Create an [initializer](https://douglasrizzo.github.io/catsim/initialization.html),
    an item [selector](https://douglasrizzo.github.io/catsim/selection.html), a
    proficiency [estimator](https://douglasrizzo.github.io/catsim/estimation.html)
    and a [stopping criterion](https://douglasrizzo.github.io/catsim/stopping.html);
4.  Pass them to a [simulator](https://douglasrizzo.github.io/catsim/simulation.html)
    and start the simulation.
5.  Access the simulator\'s properties to get specifics of the results;
6.  [Plot](https://douglasrizzo.github.io/catsim/plot.html) your results.

```python
from catsim.initialization import RandomInitializer
from catsim.selection import MaxInfoSelector
from catsim.estimation import NumericalSearchEstimator
from catsim.stopping import MaxItemStopper
from catsim.simulation import Simulator
from catsim.cat import generate_item_bank
initializer = RandomInitializer()
selector = MaxInfoSelector()
estimator = NumericalSearchEstimator()
stopper = MaxItemStopper(20)
Simulator(generate_item_bank(100), 10).simulate(initializer, selector, estimator, stopper)
```

## Dependencies

All dependencies are listed on setup.py and should be installed
automatically.

To run the tests, you\'ll need to install the testing requirements
`pip install catsim[testing]`.

To generate the documentation, Sphinx and its dependencies are needed.

## Compatibility

Since the beginning, *catsim* has only been compatible with Python 3.4
upwards.

## Important links

-   Official source code repo: <https://github.com/douglasrizzo/catsim>
-   HTML documentation (stable release):
    <https://douglasrizzo.github.io/catsim>
-   Issue tracker: <https://github.com/douglasrizzo/catsim/issues>

## Citing catsim

You can cite the package using the following bibtex entry:

```bibtex
@article{catsim,
    author = {{De Rizzo Meneghetti}, Douglas and Aquino Junior, Plinio Thomaz},
        title = "{Application and Simulation of Computerized Adaptive Tests Through the Package catsim}",
    journal = {arXiv e-prints},
    keywords = {Statistics - Applications},
        year = 2017,
        month = jul,
        eid = {arXiv:1707.03012},
        pages = {arXiv:1707.03012},
archivePrefix = {arXiv},
    eprint = {1707.03012},
primaryClass = {stat.AP}
}
```
