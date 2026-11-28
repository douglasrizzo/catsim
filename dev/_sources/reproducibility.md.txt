# Reproducibility

A simulation study is reproducible when running its code again gives the same numbers. This guide
seeds one run, says how the examinees of a run depend on each other, seeds a study of many
replications and runs a study on several processes. {doc}`running-simulations` shows how to build
and run the test itself.

## One Seed per Run

{py:func}`~catsim.simulate` takes `rng`, the source of every random draw of the run: each
examinee's answers, the initializer's starting points, the selectors' random choices and
tie-breaks, and the decisions of the engine's exposure control, if it has one. It can be:

- `None`, the default: the run draws fresh entropy, so every call gives other tests and none can be
  repeated;
- an int: a seed, so the same seed and the same examinees give the same result every time;
- a {py:class}`numpy.random.Generator`: the run draws its seed from it, which advances it, so
  passing the same generator to a second run gives other tests.

```pycon
>>> import numpy
>>> from catsim import CatEngine, simulate
>>> from catsim.estimation import NumericalSearchEstimator
>>> from catsim.initialization import UniformInitializer
>>> from catsim.selection import MaxInfoSelector
>>> from catsim.stopping import MaxLengthStopper, StandardErrorStopper, StopRule
>>> from irtlib import ItemBank
>>> item_bank = ItemBank.generate(500, rng=42)
>>> stopper = StopRule(StandardErrorStopper(0.4), MaxLengthStopper(20))
>>> engine = CatEngine(item_bank, UniformInitializer(), MaxInfoSelector(), NumericalSearchEstimator(), stopper)
>>> true_thetas = numpy.random.default_rng(7).normal(0, 1, 10)
>>> first = simulate(engine, true_thetas=true_thetas, rng=42)
>>> first.sessions == simulate(engine, true_thetas=true_thetas, rng=42).sessions
True
>>> first.sessions == simulate(engine, true_thetas=true_thetas, rng=43).sessions
False
>>> simulate(engine, true_thetas=true_thetas).sessions == simulate(engine, true_thetas=true_thetas).sessions
False
>>> generator = numpy.random.default_rng(42)
>>> simulate(engine, true_thetas=true_thetas, rng=generator).sessions == simulate(
...     engine, true_thetas=true_thetas, rng=generator
... ).sessions
False
```

A seeded run repeats on the same versions of catsim and NumPy, and across platforms its results
agree up to floating-point rounding. Seeded results can change between catsim versions, when a
strategy changes how many random numbers it draws, so record the versions with a study's seeds.

`ItemBank.generate` takes an `rng` of its own, with the same
rules, so a generated bank depends on no run. A calibrated bank read from a file needs none.

## The Population Is an Input

`simulate` never draws the examinees: their true abilities, or their recorded responses, are an
input. To simulate a population, draw the abilities with a generator of your own, as
`numpy.random.default_rng(7).normal(0, 1, 10)` does above, and pass the same array to every engine
you compare, so that every strategy meets the same examinees. Seeding the population apart from the
run also lets a study hold the examinees fixed while it varies the run's seed, or the reverse.

## Each Examinee Draws on Its Own

An examinee's answers and test depend on the run's seed and the examinee's position in the run, not
on the tests of the examinees before it:

```pycon
>>> first.sessions[:4] == simulate(engine, true_thetas=true_thetas[:4], rng=42).sessions
True
>>> first.sessions[4:] == simulate(engine, true_thetas=true_thetas[4:], rng=42).sessions
False
```

The first four examinees take the same tests in a run of their own. What counts is the position,
though: in a run without the first four, every later examinee has another position, and so other
draws. Reordering, adding or removing examinees changes the tests of those after them, so keep the
population array identical to compare one examinee across runs.

Two engines run on the same seed and examinees give every examinee the same answer to every item,
whichever items each engine chooses. Here an engine with randomesque selection gives the first
examinee 14 of the items the first engine gave, and the answers to them agree:

```pycon
>>> from catsim.selection import RandomesqueSelector
>>> other = CatEngine(item_bank, UniformInitializer(), RandomesqueSelector(bin_size=5), NumericalSearchEstimator(), stopper)
>>> second = simulate(other, true_thetas=true_thetas, rng=42)
>>> answers = dict(zip(first.sessions[0].administered_items, first.sessions[0].responses))
>>> others = dict(zip(second.sessions[0].administered_items, second.sessions[0].responses))
>>> shared = answers.keys() & others.keys()
>>> len(shared), all(answers[item] == others[item] for item in shared)
(14, True)
```

These are *common random numbers*: the difference between two strategies run on the same seed is
not blurred by examinees who happened to answer differently, so it carries less Monte Carlo error
than either strategy's own metrics ({doc}`running-simulations` compares selectors this way).

The one way a test depends on the tests before it is the exposure feedback of a selector with an
`r_max` cap: the items an examinee may be given depend on how often they were given to the
examinees before. {doc}`/adr/0013-random-numbers-and-reproducibility` explains why catsim draws
this way.

## Replications

The metrics of one run depend on its draws. To measure their spread, run each configuration
several times, each replication with a seed of its own:

```python
results = [simulate(engine, true_thetas=true_thetas, rng=seed) for seed in range(10)]
rmses = [result.rmse().value for result in results]
print(f"RMSE {numpy.mean(rmses):.3f}, standard deviation over replications {numpy.std(rmses, ddof=1):.3f}")
print(f"Monte Carlo error of the first replication's RMSE {results[0].rmse().mc_se:.3f}")
```

```text
RMSE 0.472, standard deviation over replications 0.116
Monte Carlo error of the first replication's RMSE 0.108
```

The whole study repeats from its seeds. Give every configuration you compare the same seeds, so that
the comparison uses common random numbers. Each run's metrics come with Monte Carlo errors, which
estimate this spread from the run alone: 0.108 for the first replication, against 0.116 over the
ten. Under an exposure cap the errors are approximations, and replications give the exact spread.

## Parts and Workers

`simulate(..., parts=4, workers=4)` splits the examinees, in order, into four contiguous parts of
near-equal size, and runs the parts on four processes. The result depends on the engine, the
examinees, the seed and `parts`, never on `workers`, on which process runs which part, or on
timing, so a run repeats on any machine from its seed and its number of parts. `parts=1`, the
default, is the sequential run.

Each part runs its tests in order, and counts exposure for its own examinees. Without an exposure
cap, the parts change nothing, since every examinee draws from its own streams. With a cap, an
examinee sees the exposure of the earlier examinees of its part only, so the result differs from
the sequential one; the cap still holds over the whole run. Record `parts`, which the result keeps
as `result.parts`, with the seed.

A script that runs on workers calls `simulate` under `if __name__ == "__main__":`, as every process
pool of Python requires: each worker imports the script, and without the guard each would start the
study again, which fails.

```python
# study.py
import numpy

from catsim import CatEngine, simulate
from catsim.estimation import EAPEstimator
from catsim.initialization import FixedPointInitializer
from catsim.selection import MaxInfoSelector
from catsim.stopping import MaxLengthStopper
from irtlib import ItemBank


def main() -> None:
    item_bank = ItemBank.generate(300, "3PL", rng=1)
    true_thetas = numpy.random.default_rng(2).normal(0, 1, 2000)
    for r_max in (1.0, 0.2):
        selector = MaxInfoSelector(r_max=r_max)
        engine = CatEngine(item_bank, FixedPointInitializer(0.0), selector, EAPEstimator(), MaxLengthStopper(20))
        sequential = simulate(engine, true_thetas=true_thetas, rng=3)
        in_parts = simulate(engine, true_thetas=true_thetas, rng=3, parts=4)
        on_workers = simulate(engine, true_thetas=true_thetas, rng=3, parts=4, workers=4)
        print(
            f"r_max={r_max}: parts change the tests: {in_parts.sessions != sequential.sessions}, "
            f"workers change them: {on_workers.sessions != in_parts.sessions}, "
            f"highest exposure rate {in_parts.exposure_rates().value.max():.2f}"
        )


if __name__ == "__main__":
    main()
```

```text
r_max=1.0: parts change the tests: False, workers change them: False, highest exposure rate 1.00
r_max=0.2: parts change the tests: True, workers change them: False, highest exposure rate 0.20
```

The built-in strategies run on workers from anywhere, notebooks included. A custom strategy class, a
lambda such as a custom prior, or a function defined inside another function must come from a `.py`
module that the notebook or script imports: one defined in the notebook or script itself cannot
reach the workers, and `simulate` raises before any part runs, naming what to move
({doc}`extending`). For the same reason, a script read from standard input, such as
`python - <<EOF`, cannot run on workers.

A result's `duration` is wall-clock time, which depends on `workers` and on the machine; its tests
do not.

In an application, each step of a test takes the generator of its call, and the application
chooses how its sessions draw ({doc}`embedding-catsim`).
