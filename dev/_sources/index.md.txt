# `catsim` - Computerized Adaptive Testing Toolkit

catsim computes and evaluates computerized adaptive tests based on item response theory: it
chooses each examinee's next item from an item bank, estimates the examinee's ability after each
response and decides when the test ends, and it runs simulation studies that compare the methods
for each of these steps.

:::{admonition} This version breaks code written for catsim 0.21.0 and earlier
:class: warning

That code needs changes to run, and seeded simulations give different numbers. To keep using it,
install `catsim==0.21.0`, documented at the
[0.21.0 tag](https://github.com/douglasrizzo/catsim/tree/0.21.0). The
[release notes](https://github.com/douglasrizzo/catsim/releases) list every change.
:::

## Run a simulation study

Researchers compare methods on simulated examinees: {doc}`running-simulations` builds a test, runs
it and reads its metrics, and {doc}`reproducibility` seeds a study so that it repeats.

## Write a strategy

Method authors add a method of their own: {doc}`extending` writes one strategy of each kind and
says what each must do.

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
API reference <api/index>
Architecture decisions <architecture_decisions>
contributing
references
```
