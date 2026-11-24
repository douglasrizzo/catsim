# Reproducibility

A simulation study is reproducible when running its code again gives the same numbers. This guide
shows how to seed one run, a study of many replications and a study split across processes.
{doc}`running-simulations` shows how to build and run the test itself.

## One Seed per Run

{py:meth}`SimulationRunner.run <catsim.SimulationRunner.run>` takes `rng`, the source of every
random draw of the run: the initializer's starting points, the selectors' random choices and
tie-breaks, the decisions of the engine's exposure control, if it has one, and every simulated
response. It can be:

- `None`, the default: the run draws fresh randomness, so every call gives other tests and none can
  be repeated;
- an int: a seed, so the same seed and the same true abilities give the same result every time,
  however many times the runner has run, and on any runner built with the same item bank and
  strategies;
- a {py:class}`numpy.random.Generator`: the run draws from it as it is and advances it, so passing
  the same generator to a second run gives other tests.

A seeded run repeats on the same versions of catsim and NumPy. Seeded results can change between
catsim versions, when a strategy changes how many random numbers it draws, so record the versions
with a study's seeds.

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
>>> generator = numpy.random.default_rng(42)
>>> numpy.array_equal(first, runner.run(true_thetas, rng=generator).final_thetas)
True
>>> numpy.array_equal(first, runner.run(true_thetas, rng=generator).final_thetas)
False
```

The same seed gives the same estimates, another seed or no seed other ones. A generator built from
seed 42 gives the run of seed 42 the first time, and other tests the second time, because the first
run advanced it.

{py:meth}`ItemBank.generate <catsim.ItemBank.generate>` takes an `rng` of its own, with the same
rules: pass it a seed to draw the same bank every time. A calibrated bank read from a file needs
none.

## The Population Is an Input

The runner never draws the examinees: their true abilities are an input to `run`. To simulate a
population, draw the abilities with a generator of your own, as
`numpy.random.default_rng(7).normal(0, 1, 10)` does above, and pass the same array to every engine
you compare, so that every strategy meets the same examinees. Seeding the population apart from the
run also lets a study hold the examinees fixed while it varies the run's seed, or the reverse.
Abilities from elsewhere, such as the estimates of a calibration sample, work the same way.

## Sessions Run in Sequence

A run gives its tests one examinee after another, in the order of `true_thetas`, and every session
draws from the one generator in turn. Each session's draws therefore depend on all the sessions
before it:

- reordering the examinees, or adding or removing one, changes the tests of every examinee after
  it, even under the same seed;
- with an exposure cap, the items a session may be given also depend on the items given to the
  examinees before it.

To compare one examinee's test across two runs, keep the population array identical. Why catsim
runs the sessions in sequence and draws from one generator is in
{doc}`/adr/0013-random-numbers-and-reproducibility`.

## Replications

The metrics of one run depend on its draws. To measure their spread, run each configuration several
times, each replication with a seed of its own, spawned from one root seed with
{py:class}`numpy.random.SeedSequence`:

```python
replications = numpy.random.SeedSequence(2026).spawn(10)
results = [runner.run(true_thetas, rng=numpy.random.default_rng(seed)) for seed in replications]
print(numpy.mean([result.rmse for result in results]), numpy.std([result.rmse for result in results], ddof=1))
```

The whole study repeats from the root seed, and the spawned seeds give independent streams. Give
every configuration you compare the same spawned seeds; {doc}`running-simulations` compares
selectors this way. Unseeded replications would differ from each other too, but the study could not
be repeated.

## Parallel Studies

A run's sessions do not run in parallel: their draws would then depend on which session drew first.
To use several processes, split the study into parts that do not share a generator, each with a
seed of its own. Independent replications are such parts, and so is a population split in pieces:

```python
from catsim import SimulationResult

parts = numpy.array_split(true_thetas, 2)
seeds = numpy.random.SeedSequence(2026).spawn(len(parts))
results = [runner.run(part, rng=numpy.random.default_rng(seed)) for part, seed in zip(parts, seeds)]
combined = SimulationResult(
    runner.engine.item_bank,
    sessions=[session for result in results for session in result.sessions],
    exposure_counts=sum(result.exposure_counts for result in results),
    duration=sum(result.duration for result in results),
    proposal_counts=sum(result.proposal_counts for result in results),
)
print(combined.rmse, combined.exposure_rates.max())
```

Each part depends only on its own abilities and seed, so the parts can run in separate processes,
for example with {py:class}`concurrent.futures.ProcessPoolExecutor`, in any order, and give the same
results. Building one {py:class}`~catsim.SimulationResult` from their sessions and counts gives the
metrics and exposure rates of the whole population. They differ from those of one sequential run of
the same population: each part draws from its own stream, and with an exposure cap, each part's
sessions see the exposure of that part alone.

Two fields of the combined result describe the parts, not one run. Its `duration` is the sum of the
parts' run times: compute time, longer than the wall-clock time when the parts run in parallel. Its
session ids repeat, since each run numbers its sessions from 0, so tell the parts' sessions apart by
their position in `sessions`.

In a live deployment the application builds the {py:class}`~catsim.RunContext`, and so the
generator, that its sessions draw from; {doc}`live-sessions` shows how.
