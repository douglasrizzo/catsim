# `catsim` - Computerized Adaptive Testing Toolkit

catsim runs computerized adaptive tests based on item response theory: it chooses each examinee's
next item from an item bank, estimates the examinee's ability after each response and decides when
the test ends. One engine serves two kinds of work.

## Run a simulation study

Researchers give a test to simulated examinees whose true abilities are known, and compare how well
different methods for starting the test, selecting items, estimating ability and stopping recover
those abilities. {doc}`running-simulations` builds a test, runs it on a population, reads the
results and compares strategies over replications.

## Run CAT in an application

Application developers let catsim choose the items and end the test, while their application shows
the items, collects the responses and stores the sessions. {doc}`live-sessions` runs a session item
by item, saves and resumes it, and shares item exposure across sessions.

## Find your way

- New to adaptive testing? The {doc}`introduction` explains CAT and IRT, and
  {doc}`terms-and-notation` lists the terms and symbols these pages use.
- {doc}`api/techniques/index`: each method catsim implements, with its formulas, parameters and
  sources.
- The {doc}`tutorials`: the guides as hands-on exercises, in notebooks that open in Colab.
- {doc}`reproducibility` and {doc}`extending`: seeding a study, and writing your own strategies.
- The {doc}`api/index`: every public class and function.
- The [README](https://github.com/douglasrizzo/catsim#readme): installation, a quickstart and how to
  cite catsim.

```{toctree}
:hidden: true
:maxdepth: 2

getting-started
api/techniques/index
planned-techniques
API reference <api/index>
Architecture decisions <architecture_decisions>
contributing
references
```
