# `catsim` - Computerized Adaptive Testing Toolkit

catsim computes and evaluates computerized adaptive tests based on item response theory: it
chooses each examinee's next item from an item bank, estimates the examinee's ability after each
response and decides when the test ends, and it runs simulation studies that compare the methods
for each of these steps.

:::{admonition} Coming from catsim 0.x?
:class: note

catsim 1.0 changes the API that 0.x code uses. Sessions are immutable values that the engine's
`start`, `select` and `answer` steps return; `simulate` runs studies, in parts on several processes
if asked, and its results give tables and metrics with Monte Carlo errors; several names and
arguments change, tqdm is no longer installed, and seeded simulations of generated examinees give
other numbers than 0.21 did. The
[1.0.0 release notes](https://github.com/douglasrizzo/catsim/releases/tag/1.0.0) list every change
and what to do, the {doc}`guides <getting-started>` teach the 1.0 API, and `pip install "catsim<1"`
keeps 0.x code running, with its documentation at the
[0.21.0 tag](https://github.com/douglasrizzo/catsim/tree/0.21.0).
:::

## Run a simulation study

Researchers compare methods on simulated examinees: {doc}`running-simulations` builds a test, runs
it and reads its metrics, and {doc}`reproducibility` seeds a study so that it repeats.

## Write a strategy

Method authors add a method catsim lacks: {doc}`extending` lists what each strategy receives and
returns.

## Embed catsim in an application

Developers let catsim compute each step of their application's tests: {doc}`embedding-catsim` runs
a test step by step and says what the application owns.

## Find your way

- New to adaptive testing? The {doc}`introduction` explains CAT and IRT, and
  {doc}`terms-and-notation` lists the terms and symbols.
- {doc}`api/techniques/index`: each method, with its formulas and sources.
- The {doc}`tutorials`: notebooks that open in Colab.
- The {doc}`api/index` and the {doc}`architecture_decisions`.
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
