# Extending catsim

The initializer, the selector, the estimator and the stopper of a test, and the engine's optional
exposure control, are *strategies*: objects that {py:class}`~catsim.CatEngine` calls at fixed
points of every session. A method catsim lacks is a subclass of a strategy's base class, and it
runs in the engine's steps and in {py:func}`~catsim.simulate` like the built-in ones, with their
metrics and plots. This guide is for method authors: it lists what each strategy receives and
returns and the rules every strategy keeps, then writes a custom selector and a custom stopper and
runs each in a simulation. It assumes {doc}`running-simulations`.

## The Base Classes

| Strategy | Base class | The engine calls | It returns |
|---|---|---|---|
| initializer | {py:class}`~catsim.initialization.BaseInitializer` | `initialize(*, item_bank, rng)`, once per session, before the first item | the initial ability estimate, a float |
| selector | {py:class}`~catsim.selection.BaseSelector` | `select(context)`, before each item | the index of an item neither administered nor excluded |
| estimator | {py:class}`~catsim.estimation.BaseEstimator` | `estimate(context)`, after each response | an {py:class}`~catsim.Estimate` |
| stopper | {py:class}`~catsim.stopping.BaseStopper` | `stop(context)`, after each estimate, and once before the first item | whether the test ends |
| exposure control | {py:class}`~catsim.exposure.BaseExposureControl` | `accept(context, item)`, for each item the selector proposes | whether to administer it |

The exposure control is optional: an engine has one only when it is given one, as
`CatEngine(..., exposure_control=...)`. An item it declines is excluded from the session for the
rest of its test, and the selector proposes again; the
{py:class}`~catsim.exposure.BaseExposureControl` reference shows one at work.

Decorate each method you override with `@override`, so that a type checker reports a method that
overrides nothing, such as one with a misspelt name. It comes from `typing` on Python 3.12 and later,
and from the `typing_extensions` package on Python 3.11; the code below imports it from `typing`.
Annotate arrays with the aliases of {py:mod}`catsim.typing`, `FloatArray` and `IntArray`, as
catsim's own signatures do.

## What a Strategy Reads

The initializer runs before the session has any state, so it receives only the item bank and the
generator of the call. The other strategies receive a {py:class}`~catsim.StepContext`, which the
engine builds for every call:

| Field | What it holds |
|---|---|
| `item_bank` | the session's {py:class}`~catsim.ItemBank` |
| `rng` | the generator of the call, which every random draw of the strategy comes from |
| `administered_items`, `responses` | the items given so far and the responses to them, in order, as tuples |
| `excluded_items` | items the selector must not choose although they were not given, such as those the exposure control declined |
| `theta`, `se` | the current ability estimate and its standard error |
| `exposure_rates` | each item's exposure rate, between 0 and 1, which a selector compares with its `r_max` cap, or `None`, under which no cap applies |

The selector receives the session as it stands, and the exposure control the same context with the
items it already declined in this selection among the excluded ones. The estimator receives the
newest item and response, with `theta` and `se` still those of the previous estimate, its starting
point. The stopper receives the new estimate as well; before the first item, it receives the
initial estimate, no items and an infinite `se`, so it must handle an empty test. No context holds
the examinee's true ability, so a strategy cannot see the simulated truth. A context checks its
fields when it is built, so a strategy receives checked inputs.

An estimator returns an {py:class}`~catsim.Estimate`: `theta`, which must be finite; `se`, its
standard error by the estimator's own formula, which must be positive and may be `math.inf`, since
the precision stoppers compare it with their targets; and `evaluations`, how many times the
estimator evaluated its objective, or `None`. For the likelihood, {py:func}`catsim.irt.log_likelihood_function`
returns the log-likelihood of the responses as a function of the ability, and
{py:func}`catsim.irt.log_response_probabilities` returns $\log P_i(\theta)$ and $\log Q_i(\theta)$.
Bayesian estimators can build on the priors and the posterior of
{doc}`/api/techniques/estimation/bayesian-infrastructure`.

## The Rules a Strategy Keeps

- **No per-session state.** A strategy holds its configuration and caches derived from the item
  bank, nothing else. One strategy object serves every session of a run, an application may
  interleave the sessions of many examinees on one engine, and look-ahead selectors such as MEI
  call their estimator with hypothetical responses. Whatever a strategy needs about the session, it
  computes from the context at each call.
- **Randomness only from the generator it is given:** `context.rng`, or the initializer's `rng`.
  In a simulation, that is the generator of the examinee's session, so a seeded run repeats and
  every examinee draws from streams of its own ({doc}`reproducibility`). A strategy that draws
  from anywhere else, such as a generator of its own or `numpy.random`'s global functions, escapes
  the run's seed.
- **A selector chooses an item neither administered nor excluded,** and raises
  {py:class}`~catsim.NoItemsAvailableError` when it has none to offer; the engine then ends the
  session with `StopReason.NO_ITEMS`. A selector that passes a changed context to another, with
  {py:func}`dataclasses.replace`, adds to its excluded items and never removes any.
- **A stopper is pure:** the same context always gives the same answer, so a
  {py:class}`~catsim.stopping.StopRule` can be asked after a run which of its stoppers held.
- **An estimator's estimate is finite** once an item has been administered, whatever the responses,
  all correct and all incorrect included.
- **A strategy pickles, to run on workers.** `simulate(..., workers=4)` sends the engine to other
  processes, which import each strategy's class by its module and name. Define the class in a
  module, not in a notebook or the script being run, and keep its attributes picklable: no lambdas
  or functions defined inside other functions. Otherwise `simulate` raises before any part runs,
  naming what to move. With one worker, the default, any strategy runs.

{doc}`/adr/0007-strategy-contracts` explains why the strategies are shaped this way.

## The Selector Helpers

{py:mod}`catsim.selection` exports the functions that make the built-in selectors behave alike. A
custom selector that uses them draws its candidates, applies an exposure cap and breaks ties as the
built-in ones do:

- {py:func}`~catsim.selection.remaining` returns the items neither administered nor excluded, and
  raises `NoItemsAvailableError` when there are none;
- {py:func}`~catsim.selection.below_cap` keeps the candidates whose exposure rate is below a cap
  `r_max`, or all of them when none is;
- {py:func}`~catsim.selection.pick_best` returns the candidate with the highest score, and breaks
  exact ties at random with the generator it is given, as every built-in selector does
  ({ref}`selection-ties`);
- {py:func}`~catsim.selection.best_pool` returns the best candidates, with every candidate tied at
  the cutoff, for selectors that draw from a pool of good items;
- {py:func}`~catsim.selection.check_r_max` and {py:func}`~catsim.selection.check_test_size` check a
  selector's constructor arguments.

## A Custom Selector

Content balancing keeps each test's mix of content areas, such as algebra, geometry and statistics,
close to a target. The selector below gives the most informative item of the area given least so
far in the session, among the items under an exposure cap. The areas, one per item, and their
number are configuration; how many items of each area the session has had is computed from the
context at every call, never stored.

```python
from typing import override

import numpy

from catsim import CatEngine, ItemBank, NoItemsAvailableError, StepContext, simulate
from catsim.estimation import EAPEstimator
from catsim.initialization import FixedPointInitializer
from catsim.selection import BaseSelector, MaxInfoSelector, below_cap, check_r_max, pick_best, remaining
from catsim.stopping import BaseStopper, MaxLengthStopper, StopRule
from catsim.typing import IntArray


class ContentBalancedSelector(BaseSelector):
    """Give the most informative item of the content area given least so far."""

    def __init__(self, areas: IntArray, r_max: float = 1.0) -> None:
        self.areas = numpy.asarray(areas)
        self.n_areas = int(numpy.max(self.areas)) + 1
        self.r_max = check_r_max(r_max)

    @override
    def select(self, context: StepContext) -> int:
        candidates = remaining(context)
        given = numpy.bincount(self.areas[list(context.administered_items)], minlength=self.n_areas)
        open_areas = numpy.unique(self.areas[candidates])
        area = open_areas[numpy.argmin(given[open_areas])]
        in_area = below_cap(candidates[self.areas[candidates] == area], context.exposure_rates, self.r_max)
        return pick_best(in_area, context.item_bank.information(context.theta)[in_area], context.rng)
```

Among the areas with items left, `argmin` picks the one given least, the lowest-numbered on a tie.
A strategy can be called by hand, which is how to test one, with a context built from its two
required fields and any others (for a session an engine runs, {py:meth}`~catsim.CatEngine.context`
gives the one its selector sees next):

```python
item_bank = ItemBank.generate(300, "3PL", rng=1)
areas = numpy.resize([0, 1, 2], item_bank.n_items)
selector = ContentBalancedSelector(areas, r_max=0.3)
context = StepContext(item_bank=item_bank, rng=numpy.random.default_rng(0), administered_items=(0, 1), responses=(True, False))
item = selector.select(context)
print(item, areas[item])
```

```text
155 2
```

The items belong to three areas in turn: 0, 1, 2, 0, 1, 2 and so on. Items 0 and 1 are of areas 0
and 1, so the selector gives an item of area 2, item 155. In a simulation, the selector goes
wherever a built-in one does:

```python
true_thetas = numpy.random.default_rng(2).normal(0, 1, 500)
engine = CatEngine(item_bank, FixedPointInitializer(0.0), selector, EAPEstimator(), MaxLengthStopper(12))
result = simulate(engine, true_thetas=true_thetas, rng=3)
print(areas[list(result.sessions[0].administered_items)])
print(f"RMSE {result.rmse().value:.3f}, highest exposure rate {result.exposure_rates().value.max():.2f}")
```

```text
[0 1 2 0 1 2 0 1 2 0 1 2]
RMSE 0.519, highest exposure rate 0.30
```

Every test of 12 items gives 4 of each area, in turn, and no item's exposure rate goes above the
cap of 0.3.

## A Custom Stopper

A test can end when no item left in the bank would tell much more about the examinee. The stopper
below ends the test once the most informative remaining item gives less than `min_info` at the
current estimate. It reads the item bank, the estimate and the remaining items from the context,
so it is pure.

```python
class MinInfoStopper(BaseStopper):
    """Stop once no remaining item gives at least `min_info` at the current estimate."""

    def __init__(self, min_info: float) -> None:
        self.min_info = min_info

    @override
    def stop(self, context: StepContext) -> bool:
        try:
            candidates = remaining(context)
        except NoItemsAvailableError:
            return True
        return bool(numpy.max(context.item_bank.information(context.theta)[candidates]) < self.min_info)


stopper = StopRule(MinInfoStopper(0.2), MaxLengthStopper(30), min_items=5)
engine = CatEngine(item_bank, FixedPointInitializer(0.0), MaxInfoSelector(), EAPEstimator(), stopper)
result = simulate(engine, true_thetas=true_thetas, rng=3)
early = result.examinees()["test_length"] < 30
extreme = numpy.abs(true_thetas) > 1.5
print(f"{numpy.sum(early)} tests end before 30 items")
print(f"abilities beyond ±1.5: {numpy.mean(extreme):.2f} of the population, {numpy.mean(extreme[early]):.2f} of those tests")
```

```text
39 tests end before 30 items
abilities beyond ±1.5: 0.14 of the population, 0.87 of those tests
```

A {py:class}`~catsim.stopping.StopRule` combines the custom stopper with a length cap and a minimum
length, as it does the built-in ones. In this run, 39 of the 500 tests end before the cap, nearly 9
in 10 of them for examinees whose ability lies beyond ±1.5, against 14% of the population: the bank
has few informative items at those abilities, and the stopper ends the test once it has given them.

Both strategies here are defined in the script that runs them, so they run with one worker. To run
them on several, move the two classes into a module and import them from there.
