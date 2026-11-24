# ADR-0013: Random Numbers and Reproducibility

## Status

Accepted

## Context

Three components of a simulated CAT draw random numbers: the initializer (a random starting
ability), the selectors (breaking ties, ADR-0008, and the randomized selectors' draws) and the
simulated responses, each a uniform draw against the item's probability of a correct answer. The
estimators and stoppers draw nothing.

`SimulationRunner` used to create one generator when it was constructed, from its `seed`, and pass
it to every run. Two calls to `run` with the same true abilities then gave different tests, the
second continuing the first one's stream. Its `run(n)` form also drew the `n` true abilities from
the bank's difficulties with that same generator, so the same abilities gave other tests when they
were passed explicitly instead.

Whether a run can be repeated, compared with another or split across processes depends on where the
generator comes from and on what draws from it, which nothing recorded.

## Decision

- `SimulationRunner.run(true_thetas, *, rng=0)` builds one generator,
  `numpy.random.default_rng(rng)`, at the start of every call. `rng` is a seed or a
  `numpy.random.Generator`; a generator is used as is, and the run advances it.
- That one generator serves the whole run. It reaches the strategies through `RunContext.rng` and
  `StepContext.rng` (ADR-0004) and the simulated responses through `ResponseProvider.answer`, so the
  initializer, the selectors and the responses of every session draw from one stream.
- The sessions run one after another, in the order of `true_thetas`, so the draws of a session
  depend on every draw of the sessions before it.
- The runner draws nothing itself. The true abilities are an input: to simulate a population, the
  caller draws them, e.g. `numpy.random.default_rng(1).normal(0, 1, n)`, with a generator of its
  own.
- The default `rng=0` makes a call that passes no seed deterministic. This is the decided default:
  an unseeded simulation gives the same numbers on every machine and every run, and a study that
  wants independent replications passes different seeds.

## Consequences

- The same seed and the same `true_thetas` give identical results, run after run and runner after
  runner, on the same engine configuration and package versions. A generator passed as `rng` is
  advanced by each run, so reusing the same generator object gives other results.
- Reordering `true_thetas` changes the results, and so does inserting or removing an examinee:
  every later session starts from another point of the stream.
- A change in how many numbers one component draws, such as a selector changed to break ties it
  used to leave alone, shifts the draws of every later session, so seeded results can change across
  catsim versions.
- Running the sessions of one run in parallel would make the draws depend on their timing, so the
  runner does not. To parallelise, split the population and run each part with a generator of its
  own, for example `numpy.random.default_rng(child)` for each `child` of
  `numpy.random.SeedSequence(seed).spawn(k)`, accepting that the results differ from those of one
  sequential run.
- An application that steps sessions itself (`CatEngine.step` and `apply_response`) owns its
  `RunContext` and so its generator; the same rules apply to the sessions it runs on one context.
- `ItemBank.generate` takes its own `rng`, also 0 by default, so a generated bank does not depend
  on any run.
