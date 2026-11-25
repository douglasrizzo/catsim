# ADR-0013: Every Draw Comes From a Generator the Caller Passes, One Stream per Examinee

## Status

Proposed 2026-11-25

## Context

Three components of a simulated test draw random numbers: the initializer, the selectors (ties,
ADR-0017, and the randomized selectors) with the exposure control, and the simulated responses. One
generator, built from the run's seed, served all of them for every session in turn. So examinee 7's
answers depended on how many numbers examinees 0 to 6 had drawn, two engines compared on one seed
gave the same examinee different answers to the same item, and a run could not be split across
processes without changing its results.

catR seeds `randomCAT` and `simulateRespondents` through `genSeed` and draws each response as its
item is given. mirtCAT draws each examinee's whole response pattern first, so designs compared on the
same patterns see the same answers: common random numbers, which reduce the Monte Carlo error of a
comparison.

## Decision

- Every draw comes from a generator or seed the caller passes as `rng`: `None` for fresh entropy
  (NumPy's convention), an `int` for a fixed state, or a `numpy.random.Generator`, used as is.
  catsim never uses global random state.
- Each step method takes the generator of its call (ADR-0006). The application chooses how its
  sessions draw.
- `simulate` derives one root entropy from `rng`: the seed itself, or a number drawn once from a
  passed generator. Examinee `i`, at position `i` of the run's examinees, draws from two streams of
  its own:
  - `SeedSequence(entropy, spawn_key=(i, 0))` for its response pattern (ADR-0014);
  - `SeedSequence(entropy, spawn_key=(i, 1))` for its session: the initializer, the selector and
    the exposure control.
- `ItemBank.generate` takes its own `rng`, so a generated bank depends on no run.

## Consequences

- The same seed, examinees and configuration give the same output, on the same versions of catsim
  and NumPy. Across platforms results agree up to floating-point rounding.
- A session depends on other sessions only through the exposure feedback (ADR-0012). Without it, a
  run can be split anywhere and gives the same output (ADR-0015).
- Two engines run on the same seed and examinees give every examinee the same answer to every item,
  so a comparison between them carries less Monte Carlo error.
- Reordering, inserting or removing examinees changes the streams of those at later positions.
- An unseeded run cannot be repeated, and a passed generator gives other results each time, since
  the run draws from it.
- A component that draws one more number changes only its own session's later draws.

## Alternatives Considered

- One generator for the whole run, in session order: comparisons lose common random numbers, and
  splitting a run changes its results.
- Responses drawn as items are given, from the examinee's stream: two engines that give items in
  different orders would give the same examinee different answers to the same item.
- A fixed default seed: unseeded replications in a loop are identical, against NumPy's convention.
- Streams keyed by an examinee identifier: a run's examinees are its true abilities or its response
  rows, which have no identifier but their position.
