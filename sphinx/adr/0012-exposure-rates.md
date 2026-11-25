# ADR-0012: Exposure Rates Are an Input, Counted as catR Counts Them

## Status

Proposed 2026-11-25

## Context

Selectors with an `r_max` cap skip items whose exposure rate has reached it. The item bank once kept
a column of exposure rates, so a bank reused across runs carried one run's exposure into the next.
Later an `ExposureTracker` on the run context counted administrations, and the engine updated it;
an application had to share one tracker across its live sessions and save it across restarts.

The rates were first the counts divided by the planned size of the run, so early examinees saw every
rate near zero, and without the planned size an `r_max` cap silently did nothing. catR's
`simulateRespondents` divides by the examinees tested so far.

## Decision

- Exposure rates are an explicit input of `select`: an array with one rate per item, each in
  $[0, 1]$, or `None`, under which no cap applies. catsim keeps no exposure counts between calls.
- An application computes the rates from the counts it keeps, by whatever rule suits sessions that
  run side by side.
- `simulate` keeps the counts of a run, or of each part of a parallel run (ADR-0015), and computes
  catR's rate: each item's administrations divided by the number of sessions started before the
  current one. The first session sees no rates.
- An `r_max` of 1 applies no cap, as in catR: an item every earlier examinee received has a rate of
  exactly 1, which a strict `rate < r_max` test would pass over.
- `SimulationResult.exposure_rates` divides the run's counts by all its sessions: the exposure at
  the end of the run, not the rates any session saw.

## Consequences

- An item bank can be reused across sessions, runs and processes, and its cached values never go
  stale.
- A live deployment applies a cap by passing rates, without knowing how many examinees it will test,
  and shares exposure across sessions by sharing its counts, which is its own concern.
- A simulation's exposure feedback is sequential: each session depends on the ones before it in its
  part.

## Alternatives Considered

- Rates computed by catsim from a tracker the caller keeps: catsim's state again, outliving each
  call, which the application must share and save.
- Dividing by the planned size of the run: it leaves a live deployment without a cap and makes the
  cap bear on examinees by their place in the run.
- Counting the current session in the denominator: it lowers the early sessions' rates most, so the
  second session would see the first one's items at 1/2, not 1.
- Exposure kept on the item bank: one run's exposure would leak into the next.
