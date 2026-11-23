# Reproducibility

Objects that use random number generation directly, mainly initializers and selectors, can have reproducible outputs by receiving a {py:class}`numpy.random.Generator` instance in the `rng` keyword argument of their main method.

In the snippet below, all selectors that have random behavior produce the same outputs, when given the same input arguments.

```{code-block} python
:caption: Generating reproducible outputs from CAT components.

from catsim import ItemBank
from catsim.selection import RandomesqueSelector, RandomSelector, The54321Selector
from numpy.random import default_rng

for _ in range(5):
    item_bank = ItemBank.generate_item_bank(5000, seed=42)
    print(
        RandomSelector().select(
            item_bank=item_bank, administered_items=[], rng=default_rng(42), est_theta=0
        ),
        The54321Selector().select(
            item_bank=item_bank, administered_items=[], rng=default_rng(42), est_theta=0
        ),
        RandomesqueSelector(bin_size=10).select(
            item_bank=item_bank, administered_items=[], rng=default_rng(42), est_theta=0
        ),
    )
```

A {py:class}`catsim.simulation.SimulationRunner` creates one {py:class}`numpy.random.Generator` from its `seed` argument when it is built. `seed` defaults to 0, so a runner built without a seed is deterministic too. Every random draw in a simulation comes from that one generator: sampling the examinees' abilities, the initializer, the selectors and the simulated responses. Sessions run one after another, in order, so each session's draws depend on the sessions before it.

Two runners built with the same seed and the same components give identical results. Each call to `run()`, however, continues the runner's generator instead of restarting it, so a second `run()` on the same runner gives different results. To repeat a simulation exactly, build a new runner.

```pycon
>>> from catsim import ItemBank
>>> from catsim.estimation import NumericalSearchEstimator
>>> from catsim.initialization import RandomInitializer
>>> from catsim.selection import MaxInfoSelector
>>> from catsim.simulation import SimulationRunner
>>> from catsim.stopping import MinErrorStopper
>>> item_bank = ItemBank.generate_item_bank(500, seed=42)
>>> def make_runner():
...     return SimulationRunner(
...         item_bank,
...         RandomInitializer(),
...         MaxInfoSelector(),
...         NumericalSearchEstimator(),
...         MinErrorStopper(0.4, max_items=20),
...         seed=42,
...     )
>>> first, second = make_runner().run(10), make_runner().run(10)
>>> first.latest_estimations == second.latest_estimations
True
>>> runner = make_runner()
>>> runner.run(10).latest_estimations == runner.run(10).latest_estimations
False
```
