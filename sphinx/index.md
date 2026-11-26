# `catsim` - Computerized Adaptive Testing Toolkit

catsim computes and evaluates computerized adaptive tests based on item response theory: it
chooses each examinee's next item from an item bank, estimates the examinee's ability after each
response and decides when the test ends, and it runs simulation studies that compare the methods
for each of these steps.

:::{admonition} catsim 1.0 breaks 0.x code
:class: warning

Code written for catsim 0.x needs changes to run on 1.0, and seeded simulations give different
numbers. To keep using 0.x, install `catsim<1`, documented at the
[0.21.0 tag](https://github.com/douglasrizzo/catsim/tree/0.21.0). The
[1.0.0 release notes](https://github.com/douglasrizzo/catsim/releases/tag/1.0.0) list every change.
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
planned-techniques
API reference <api/index>
Architecture decisions <architecture_decisions>
contributing
references
```
