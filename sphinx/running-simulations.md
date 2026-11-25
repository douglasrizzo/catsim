# Running Simulations

A simulation study gives a computerized adaptive test to simulated examinees whose true abilities
are known, and compares the final ability estimates with those abilities. This guide builds a test,
runs it on a simulated population, reads the results, compares strategies over replications and
plots them. The {doc}`introduction` explains the CAT loop, and {doc}`terms-and-notation` lists the
terms used here.

## Build the Test

A test is an item bank and four strategies: an *initializer* gives the first ability estimate, a
*selector* chooses each item, an *estimator* updates the estimate after each response and a
*stopper* ends the test. A {py:class}`~catsim.CatEngine` holds them.

```python
import numpy

from catsim import CatEngine, ItemBank, SimulationRunner
from catsim.estimation import EAPEstimator
from catsim.initialization import FixedPointInitializer
from catsim.selection import MaxInfoSelector
from catsim.stopping import MaxLengthStopper, StandardErrorStopper, StopRule

item_bank = ItemBank.generate(300, "3PL", rng=1)
stopper = StopRule(StandardErrorStopper(0.35), MaxLengthStopper(30))
engine = CatEngine(item_bank, FixedPointInitializer(0.0), MaxInfoSelector(), EAPEstimator(), stopper)
```

{py:meth}`ItemBank.generate <catsim.ItemBank.generate>` draws 300 items of the three-parameter
logistic model, and its seed gives the same bank every time. For a calibrated bank, pass its
parameters to {py:class}`~catsim.ItemBank` as an array with one row per item and the columns $a$,
$b$, $c$ and $d$, for example `ItemBank(numpy.loadtxt("items.csv", delimiter=","))`.

Each strategy is a choice of method. The technique pages define each method and say when to choose
it: {doc}`initializers </api/techniques/initialization/index>`,
{doc}`selectors </api/techniques/selection/index>`,
{doc}`estimators </api/techniques/estimation/index>` and
{doc}`stoppers </api/techniques/stopping/index>`.

The stopper above ends a test once the standard error of the ability estimate is at most 0.35, or
after 30 items, whichever comes first ({ref}`combining-stopping-rules`). Keep a length cap in every
variable-length test: a precision target that the bank cannot reach for some examinees would
otherwise give them every item in the bank. With the bank and population of this guide, a target
of 0.3 runs every test to the cap.

## Run It on a Population

```python
true_thetas = numpy.random.default_rng(2).normal(0, 1, 500)
result = SimulationRunner(engine).run(true_thetas, rng=3)
```

The runner gives one test to each true ability, one examinee after another, in the order of
`true_thetas`. It never draws the abilities itself: draw them with a generator of your own, as
above, so that the same population can be passed to every engine you compare. `progress=True`
shows a progress bar over the examinees.

Each simulated response is a Bernoulli draw: correct with the probability that the item's
characteristic curve gives at the examinee's true ability, $P_i(\theta)$
({py:func}`catsim.irt.icc`), decided by one uniform random draw from the run's generator.

`rng` is the source of every random draw of the run: the initializer's, the selector's, those of
the engine's exposure control, if it has one, and the simulated responses. An int seeds it, so the
same seed and the same true abilities give the same result every time. None, the default, draws
fresh randomness, so two unseeded runs give different tests. A {py:class}`numpy.random.Generator` is
used as is, and the run advances it. {doc}`reproducibility` explains how to seed a whole study.

## Read the Result

`run` returns a {py:class}`~catsim.SimulationResult`, which holds the sessions, in the order of
`true_thetas`, and summaries over them.

```python
print(f"bias {result.bias:.3f}, RMSE {result.rmse:.3f}")
print(f"mean test length {result.test_lengths.mean():.1f}")
print(f"share above the target {numpy.mean(result.final_ses > 0.35):.2f}")
print(f"highest exposure rate {result.exposure_rates.max():.2f}")
print(f"overlap rate {result.overlap_rate}")
```

| Attribute | What it holds |
|---|---|
| `bias`, `mse`, `rmse` | how far the final estimates are from the true abilities, as floats; {py:mod}`catsim.metrics` gives the formulas |
| `final_thetas`, `final_ses`, `true_thetas` | each examinee's final estimate, its standard error and the true ability, as arrays |
| `test_lengths` | how many items each examinee was given |
| `exposure_rates` | the share of examinees given each item |
| `proposal_rates` | the share of examinees for whom the selector proposed each item; it equals `exposure_rates` unless the engine has an exposure control |
| `overlap_rate` | the expected share of items two examinees have in common, or None |
| `evaluations`, `mean_evaluations` | how many times the estimator evaluated its objective, a measure of its cost that does not depend on the machine |
| `duration` | the run's wall-clock time, in seconds |
| `sessions` | each examinee's {py:class}`~catsim.CatSessionState`: `administered_items`, `responses`, `theta_history` and `stop_reason` |

In this run the bias is close to 0 and the RMSE close to the 0.35 target. Tests take 25 items on
average, and for 10% of the examinees the test ended at the 30-item cap with a standard error still
above 0.35: the bank has too few informative items at their abilities. The highest exposure rate is
1, because every test starts at the same estimate and so with the same item.

The overlap rate prints None: its formula assumes that every examinee takes the same number of
items, so it is defined only for fixed-length tests, such as those of a `MaxLengthStopper` alone.

With an estimator that maximizes the likelihood, such as `NumericalSearchEstimator`, the
likelihood of a short test's responses can keep rising toward the edge of the ability scale, which
catsim keeps within $[-6, 6]$. A few examinees then end with an estimate of $-6$ or $6$, and their
errors dominate the RMSE of the run. A Bayesian estimator, such as the EAP above, keeps estimates
near its prior ({doc}`/adr/0018-ability-scale-and-extreme-patterns`).

## Compare Strategies over Replications

The bias and RMSE of one run depend on its random draws. To compare strategies, run each over
several replications, on the same item bank and the same population, with the same seeds for every
strategy, and compare the strategies replication by replication.

```python
from catsim.selection import RandomesqueSelector

selectors = {
    "max info": MaxInfoSelector(),
    "max info, r_max=0.2": MaxInfoSelector(r_max=0.2),
    "randomesque": RandomesqueSelector(bin_size=5),
}
replications = numpy.random.SeedSequence(2026).spawn(5)
rmse = {}
for name, selector in selectors.items():
    runner = SimulationRunner(CatEngine(item_bank, FixedPointInitializer(0.0), selector, EAPEstimator(), stopper))
    results = [runner.run(true_thetas, rng=numpy.random.default_rng(seed)) for seed in replications]
    rmse[name] = numpy.array([replication.rmse for replication in results])
    length = numpy.mean([replication.test_lengths.mean() for replication in results])
    exposure = numpy.mean([replication.exposure_rates.max() for replication in results])
    print(f"{name}: RMSE {rmse[name].mean():.3f}, length {length:.1f}, top exposure {exposure:.2f}")

pairs = [("max info, r_max=0.2", "randomesque"), ("max info", "randomesque"), ("max info", "max info, r_max=0.2")]
for first, second in pairs:
    difference = rmse[first] - rmse[second]
    standard_error = numpy.std(difference, ddof=1) / numpy.sqrt(len(difference))
    print(f"{first} minus {second}: RMSE {difference.mean():+.3f} ± {standard_error:.3f}")
```

Each replication has a seed of its own, spawned from one root seed: the whole study repeats from
that seed, and every selector meets the same five seeds. Without `rng`, the replications would still
differ from each other, but the study could not be repeated.

The output shows the trade-off between precision and exposure. The cap `r_max=0.2` keeps every item's
rate at or below 0.2, at the cost of about three more items per test. Randomesque selection, which
draws each item from the five most informative, lowers the highest rate to about 0.8. Each
replication gives the selectors the same population and seed, so the last lines compare them by
their difference in each replication: its mean, and the standard error of that mean. In this run, the
cap's RMSE is above randomesque's by 0.020 ± 0.005, four standard errors. The gaps of
about 0.01 between maximum information and the other two are within about two standard errors:
ranking those would take more replications or more examinees. The selection page compares the
{doc}`exposure-control methods </api/techniques/selection/index>`. A cap is checked against the
share of the examinees tested before the current one, so it applies from the second examinee on
({py:class}`~catsim.ExposureTracker`).

## Plot the Results

The plots need matplotlib, which the plot extra installs: `pip install 'catsim[plot]'`.

```python
import matplotlib.pyplot as plt

from catsim import plot

plot.session_progress(result.sessions[0], result.item_bank)
plot.item_exposure(result.exposure_rates, item_bank=result.item_bank, par="b")
plt.show()
```

The first figure follows one examinee: the ability estimate after each item, with a band of one
standard error on each side, the difficulty of each item given and the true ability. The band
narrows as items are given, and this examinee's test ends when the standard error reaches the
target. The second figure draws each item's exposure rate at its difficulty. In this run, the items
given most are among the most discriminating, with difficulties near the middle of the population,
and more than half of the bank is never given. {py:mod}`catsim.plot` lists the other plots.
