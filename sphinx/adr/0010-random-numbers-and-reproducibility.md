# ADR-0010: Random Numbers and Reproducibility

## Status

Accepted 2026-11-24

## Context

Three components of a simulated CAT draw random numbers: the initializer (a random starting
ability), the selectors (breaking ties, ADR-0011, and the randomized selectors' draws) and the
simulated responses, each a uniform draw against the item's probability of a correct answer. The
estimators and stoppers draw nothing.

`SimulationRunner` created one generator from its `seed` when it was constructed and passed it to
every run, and its `run(n)` form drew the true abilities from the bank's difficulties with that
same generator. Later, `run` and `ItemBank.generate` defaulted to `rng=0`, so every unseeded call
gave the same run or bank. NumPy, `scipy.stats` and scikit-learn read a missing seed as fresh
randomness.

Whether a run can be repeated, compared with another or split across processes depends on where the
generator comes from and on what draws from it, which nothing recorded.

## Decision

- `SimulationRunner.run(true_thetas, *, rng=None)` builds one generator,
  `numpy.random.default_rng(rng)`, at the start of every call. `rng` is None, a seed or a
  `numpy.random.Generator`: None draws fresh entropy (NumPy's convention), a seed gives a generator
  in a fixed state, and a generator is used as is and advanced by the run.
- That one generator serves the whole run. It reaches the strategies through `RunContext.rng` and
  `StepContext.rng` (ADR-0005) and the simulated responses through `ResponseProvider.answer`, so the
  initializer, the selectors, the engine's exposure control (ADR-0008) and the responses of every
  session draw from one stream. An engine without an exposure control draws nothing for it.
- The sessions run one after another, in the order of `true_thetas`, so the draws of a session
  depend on every draw of the sessions before it.
- The runner draws nothing itself. The true abilities are an input: to simulate a population, the
  caller draws them with a generator of its own.
- Reproducibility comes from passing `rng`. An unseeded call is random, so unseeded replications in
  a loop differ, as they must for their spread to mean anything.

## Consequences

- The same seed and the same `true_thetas` give identical results, run after run and runner after
  runner, on the same engine configuration and package versions; a generator passed as `rng` gives
  other results each time, since each run advances it.
- An unseeded run cannot be repeated: nothing records the entropy it drew.
- Reordering `true_thetas` changes the results, and so does inserting or removing an examinee:
  every later session starts from another point of the stream.
- A change in how many numbers one component draws, such as a selector changed to break ties it
  used to leave alone, shifts the draws of every later session, so seeded results can change across
  catsim versions.
- Running the sessions of one run in parallel would make the draws depend on their timing, so the
  runner does not. A parallel study splits the population and runs each part with a generator of
  its own, and its results differ from those of one sequential run ({doc}`/reproducibility`).
- An application that steps sessions itself (`CatEngine.step` and `apply_response`) owns its
  `RunContext` and so its generator; the same rules apply to the sessions it runs on one context.
- `ItemBank.generate` takes its own `rng`, also None by default, so a generated bank does not
  depend on any run.

## Alternatives Considered

- One generator per runner, created when the runner is built: two runs on the same true abilities
  give different tests, the second continuing the first one's stream.
- A fixed default seed: unseeded replications in a loop are identical, with no warning, against
  the convention of NumPy and SciPy.
- Drawing the true abilities inside the runner: the same abilities give other tests depending on
  whether they were drawn or passed in.
