# Reproducibility

Objects that use random number generation directly, mainly initializers, selectors and exposure controls, can have reproducible outputs by receiving a {py:class}`numpy.random.Generator` instance: initializers take it as the `rng` keyword argument of `initialize`, and selectors and exposure controls read it from the `rng` field of the {py:class}`~catsim.StepContext` passed to `select` or `accept`.

In the snippet below, all selectors that have random behavior produce the same outputs, when given the same input arguments.

```{code-block} python
:caption: Generating reproducible outputs from CAT components.

from catsim import ItemBank, StepContext
from catsim.selection import RandomesqueSelector, RandomSelector, The54321Selector
from numpy.random import default_rng

for _ in range(5):
    item_bank = ItemBank.generate(5000, rng=42)
    print(
        RandomSelector().select(StepContext(item_bank=item_bank, rng=default_rng(42))),
        The54321Selector().select(StepContext(item_bank=item_bank, rng=default_rng(42))),
        RandomesqueSelector(bin_size=10).select(StepContext(item_bank=item_bank, rng=default_rng(42))),
    )
```

{py:meth}`SimulationRunner.run <catsim.SimulationRunner.run>` builds one {py:class}`numpy.random.Generator` from its `rng` argument at the start of every call, and every random draw of the run comes from it: the initializer, the selectors, the engine's exposure control, if it has one, and the simulated responses of every session ({doc}`/adr/0013-random-numbers-and-reproducibility`). Results are reproducible when `rng` is passed. Without it, the run draws fresh randomness, so every call gives other tests: replications in a loop differ, as they should, but no such run can be repeated. `rng` can also be a generator, which the run uses as is and advances. {py:meth}`ItemBank.generate <catsim.ItemBank.generate>` takes an `rng` of its own, with the same rules, so pass a seed there too to draw the same bank every time.

The runner draws nothing else. The examinees' true abilities are an input to `run`: to simulate a population, draw them with a generator of your own, as `rng.normal(0, 1, n)` does below, so the same abilities can be run again, on another engine or with another seed.

The same seed and the same true abilities give identical results, however many times the runner has run and on any runner built with the same components. A different seed, or no `rng` at all, gives other tests. A generator passed as `rng` is not a seed: each run advances it, so passing the same generator object again gives other tests.

```pycon
>>> import numpy
>>> from catsim import CatEngine, ItemBank, SimulationRunner
>>> from catsim.estimation import NumericalSearchEstimator
>>> from catsim.initialization import UniformInitializer
>>> from catsim.selection import MaxInfoSelector
>>> from catsim.stopping import MaxLengthStopper, StandardErrorStopper, StopRule
>>> engine = CatEngine(
...     ItemBank.generate(500, rng=42),
...     UniformInitializer(),
...     MaxInfoSelector(),
...     NumericalSearchEstimator(),
...     StopRule(StandardErrorStopper(0.4), MaxLengthStopper(20)),
... )
>>> true_thetas = numpy.random.default_rng(7).normal(0, 1, 10)
>>> runner = SimulationRunner(engine)
>>> first = runner.run(true_thetas, rng=42).final_thetas
>>> numpy.array_equal(first, runner.run(true_thetas, rng=42).final_thetas)
True
>>> numpy.array_equal(first, runner.run(true_thetas, rng=43).final_thetas)
False
>>> numpy.array_equal(runner.run(true_thetas).final_thetas, runner.run(true_thetas).final_thetas)
False
```

Sessions run one after another, in the order of `true_thetas`, so each session's draws depend on the sessions before it. Reordering the examinees, or adding or removing one, changes the tests of the sessions after it, and the sessions of one run cannot run in parallel without changing the results. To parallelise a study, split the population and run each part with a generator of its own, such as `numpy.random.default_rng(child)` for each `child` of `numpy.random.SeedSequence(42).spawn(4)`, and accept that the results differ from those of one sequential run.
