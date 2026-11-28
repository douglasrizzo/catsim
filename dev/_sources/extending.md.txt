# Extending catsim

A method catsim lacks is a class: subclass the base class of its kind of strategy and implement its
one method. {py:class}`~catsim.CatEngine` then calls it at fixed points of every test, and it runs
in {py:func}`~catsim.simulate` like the built-in strategies, with their metrics and plots. This
guide writes an initializer, an estimator, a selector and a stopper, says after each what that kind
of strategy must do, and ends by running all four in one simulation. It assumes
{doc}`running-simulations`.

The examples share these imports:

```python
from typing import override

import numpy

from catsim import CatEngine, Estimate, NoItemsAvailableError, StepContext, simulate
from catsim.estimation import BaseEstimator
from catsim.initialization import BaseInitializer
from catsim.selection import BaseSelector, below_cap, check_r_max, pick_best, remaining
from catsim.stopping import BaseStopper, MaxLengthStopper, StopRule
from irtlib import ItemBank
from irtlib.irt import log_likelihood
from irtlib.typing import IntArray
```

The item bank, the item response functions and the array types come from
[irtlib](https://github.com/douglasrizzo/irtlib), the item response theory package catsim builds
on.

`@override` marks each method that implements one of a base class, so that a type checker reports a
method that overrides nothing, such as one with a misspelt name.

## An Initializer

An initializer gives each session its ability estimate before the first item. When every examinee
starts at the same estimate, every examinee gets the same first item. This initializer starts each
session near the bank's median difficulty, moved at random by up to `spread`:

```python
class MedianDifficultyInitializer(BaseInitializer):
    """Start near the bank's median difficulty, moved at random by up to `spread`."""

    def __init__(self, spread: float = 0.5) -> None:
        self.spread = spread

    @override
    def initialize(self, *, item_bank: ItemBank, rng: numpy.random.Generator) -> float:
        median = float(numpy.median(item_bank.difficulty))
        return median + rng.uniform(-self.spread, self.spread)
```

**What an initializer must do:** implement
{py:meth}`~catsim.initialization.BaseInitializer.initialize`, which the engine calls once per
session, before any item. The session has no responses yet, so the initializer receives only the
item bank and the generator of the call, and it returns the initial estimate as a finite float.

This initializer also keeps the three rules that every strategy keeps:

- **It draws only from the generator it is given:** `rng` here, `context.rng` in the other
  strategies. In a simulation, that generator is the examinee's own, so a seeded run repeats
  ({doc}`reproducibility`). A generator of the strategy's own, or `numpy.random`'s global
  functions, escapes the run's seed.
- **It keeps nothing about a session.** A strategy holds its configuration, and data it derives from
  the item bank, nothing else: one strategy object serves every session of a run, and an
  application may interleave the sessions of many examinees on one engine. Whatever a strategy needs
  about the session, it computes at each call.
- **It can run on workers,** so that `simulate(..., workers=4)` can send it to other processes:
  define the class in a module, not in a notebook or in the script being run, and give it no lambdas
  or functions defined inside other functions as attributes. Otherwise `simulate` raises before it
  runs anything, and names what to move. With one worker, the default, any strategy runs.

{doc}`/adr/0007-strategy-contracts` explains why strategies are shaped this way.

## An Estimator

An estimator updates the ability estimate after each response. This one computes the maximum a
posteriori estimate under a standard normal prior, by brute force: it evaluates the log-posterior on
a grid of abilities and returns the grid point where it is highest. Its standard error is
$1/\sqrt{I(\hat{\theta}) + 1}$, where $I(\hat{\theta})$ is the test information at the estimate and
the 1 comes from the prior.

```python
class GridMAPEstimator(BaseEstimator):
    """Return the most probable ability on a grid, under a standard normal prior."""

    def __init__(self, low: float = -4.0, high: float = 4.0, points: int = 161) -> None:
        self.grid = numpy.linspace(low, high, points)

    @override
    def estimate(self, context: StepContext) -> Estimate:
        items = context.item_bank.get_items(context.administered_items)
        # one row of responses, the session's, gives one row of log-likelihoods
        log_posterior = log_likelihood(self.grid, [context.responses], items)[0] - self.grid**2 / 2
        theta = float(self.grid[numpy.argmax(log_posterior)])
        information = context.item_bank.total_information(theta, context.administered_items)
        return Estimate(theta=theta, se=1 / numpy.sqrt(information + 1), evaluations=self.grid.size)
```

The estimator, the selector and the stopper receive a {py:class}`~catsim.StepContext`: what they
may read about the session, built anew by the engine for every call.

| Field | What it holds |
|---|---|
| `item_bank` | the session's `ItemBank` |
| `administered_items`, `responses` | the items given so far and the responses to them, in order |
| `excluded_items` | the items the session must not be given, although it has not had them |
| `theta`, `se` | the current ability estimate and its standard error |
| `exposure_rates` | each item's exposure rate, from 0 to 1, or `None` when no rates are given |
| `rng` | the generator of the call |

No context holds the examinee's true ability, so no strategy can see the simulated truth.

**What an estimator must do:** implement {py:meth}`~catsim.estimation.BaseEstimator.estimate`,
which the engine calls after each response. Its context holds the newest item and response, while
`theta` and `se` are still the previous estimate's, from which an iterative estimator can start. It
returns an {py:class}`~catsim.Estimate` of three fields:

- `theta`, finite whatever the responses, all correct and all incorrect included; a grid has no
  infinite points, so this estimator's estimate is always finite;
- `se`, the standard error by the estimator's own formula: positive, possibly `math.inf`. The
  stoppers that target a precision compare it with their targets;
- `evaluations`, how many times the estimator evaluated its objective, or `None`.

Look-ahead selectors, such as MEI, also call the estimator with responses the examinee has not
given, which is one more reason for it to keep nothing about a session. For the likelihood,
`irtlib.irt.log_likelihood_function` returns the log-likelihood of each row of responses as a
function of the ability, and `irtlib.irt.log_response_probabilities` returns
$\log P_i(\theta)$ and $\log Q_i(\theta)$. A Bayesian estimator can build on irtlib's quadrature
grid, priors and posterior, in `irtlib.quadrature`.

## A Selector

A selector chooses the next item. Content balancing keeps each test's mix of content areas, such as
algebra, geometry and statistics, close to a target. This selector gives the most informative item
of the area the session has had least, among the items under an exposure cap. The item's areas are
configuration; how many items of each area the session has had, it counts from the context at every
call.

```python
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

Among the areas with items left, `argmin` picks the area given least, the lowest-numbered on a tie.
The helpers of {py:mod}`catsim.selection` make a custom selector behave like the built-in ones:

- {py:func}`~catsim.selection.remaining` returns the candidate items, those neither administered nor
  excluded, and raises {py:class}`~catsim.NoItemsAvailableError` when there are none;
- {py:func}`~catsim.selection.below_cap` keeps the candidate items whose exposure rate is below a cap
  `r_max`, or all of them when none is, and treats `None` rates as no exposure;
- {py:func}`~catsim.selection.pick_best` returns the candidate item with the highest score, and
  breaks exact ties at random with the generator it is given ({ref}`selection-ties`);
- {py:func}`~catsim.selection.best_pool` returns the best candidate items, for a selector that draws
  from a pool of good ones;
- {py:func}`~catsim.selection.check_r_max` and {py:func}`~catsim.selection.check_test_size` check a
  selector's settings.

**What a selector must do:** implement {py:meth}`~catsim.selection.BaseSelector.select`, which the
engine calls before each item. It returns the index of an item neither administered nor excluded,
or raises `NoItemsAvailableError` when it has none to offer, as `remaining` does for it; the engine
then ends the session with `StopReason.NO_ITEMS`. A selector that calls another with a changed
context, made with {py:func}`dataclasses.replace`, may add excluded items but never removes one.

A strategy can be called by hand, which is how to test one. A context needs only an item bank and a
generator; its other fields default to an empty test. For a session that an engine runs,
{py:meth}`CatEngine.context <catsim.CatEngine.context>` returns the context its selector sees next.

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

The items belong to the three areas in turn: 0, 1, 2, 0, 1, 2 and so on. Items 0 and 1 are of areas
0 and 1, so the selector gives an item of area 2, item 155.

## A Stopper

A stopper decides whether the test ends. A test can end when no item left in the bank would tell
much more about the examinee. This stopper ends it once the most informative remaining item gives
less than `min_info` at the current estimate:

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
```

**What a stopper must do:** implement {py:meth}`~catsim.stopping.BaseStopper.stop`, which the
engine calls after each estimate, and once before the first item. That first call has no items and
an infinite `se`, so a stopper must handle an empty test. It returns whether the test ends, and it
is pure: the same context always gives the same answer, so a
{py:class}`~catsim.stopping.StopRule` can tell after a run which of its stoppers held.

## All Four in One Test

The four strategies go wherever built-in ones do. Here a {py:class}`~catsim.stopping.StopRule`
combines the custom stopper with a length cap of 30 items and a minimum of 5, as it combines
built-in stoppers:

```python
engine = CatEngine(
    item_bank,
    MedianDifficultyInitializer(spread=0.5),
    selector,
    GridMAPEstimator(),
    StopRule(MinInfoStopper(0.2), MaxLengthStopper(30), min_items=5),
)
true_thetas = numpy.random.default_rng(2).normal(0, 1, 500)
result = simulate(engine, true_thetas=true_thetas, rng=3)
lengths = result.examinees()["test_length"]
first_items = {session.administered_items[0] for session in result.sessions}
print(areas[list(result.sessions[0].administered_items[:9])])
print(f"{len(first_items)} first items, highest exposure rate {result.exposure_rates().value.max():.2f}")
print(f"{numpy.sum(lengths < 30)} tests end before 30 items; RMSE {result.rmse().value:.3f}")
```

```text
[0 1 2 0 1 2 0 1 2]
26 first items, highest exposure rate 0.30
8 tests end before 30 items; RMSE 0.358
```

Each test takes its items from the three areas in turn, as the first nine items of the first test
show. The initializer spreads the first items over 26 items of the bank, and the selector's cap
keeps every item's exposure rate at 0.3 or below. The 8 tests that end before the cap are all of
examinees whose ability lies beyond ±1.9: the bank has few informative items there, and the stopper
ends the test once it has given them.

An exposure control is written the same way: subclass
{py:class}`~catsim.exposure.BaseExposureControl` and implement `accept(context, item)`, which says
whether to administer the item the selector proposes. The
[custom-strategies notebook](https://colab.research.google.com/github/douglasrizzo/catsim/blob/main/notebooks/04_custom_strategies.ipynb)
writes one, along with another selector and stopper.

The strategies of this guide are defined in the script that runs them, so they run with one worker.
To run them on several, move the classes into a module and import them from there.
