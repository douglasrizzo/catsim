---
title: Correlation, test-length spread and a one-row run summary
kind: feature
status: ready
priority: 2
effort: S
blocked_by: []
raised_by: [Babcock & Weiss (2012) walkthrough]
---

# Correlation, test-length spread and a one-row run summary

Found by the Babcock & Weiss (2012) replication of the research walkthrough ([discovery](../discovery/README.md)).

## Problem

A stopping-rule study reports, per condition, the correlation r(θ, θ̂), the mean test length and its
SD, minimum and maximum, and the share of tests that reached the maximum length; overall and
conditional on θ. catsim has none of these but the mean length. The agent wrote a `marginal_row()`
helper and computed each by hand, for 128 runs.

## Evidence

- No correlation in `catsim.metrics` (`mean`, `bias`, `mse`, `rmse`, `exposure_rates`,
  `overlap_rate`).
- `deciles()` and `ability_levels()` have `test_length` (the mean) only.
- On the peaked 100-item bank, 87–99% of examinees with |θ| ≥ 2 ran to the 50-item cap under the SE
  rules: the "% at cap" column is how the paper shows that a bank cannot meet an SE target
  (the walkthrough's `out/tables.md` (kept locally), Table 6).

## Proposed shape (for the implementer to settle)

- `metrics.correlation(true_thetas, thetas)` with an MC error (Fisher z interval or its delta-method
  SE; cite the source).
- Length SD, minimum and maximum in `deciles()` and `ability_levels()`, and a share-at-maximum only if
  it can be defined without knowing the stopper (the maximum observed length is not the cap; maybe the
  share of sessions with the run's maximum length, or leave it to table-columns's stop-criterion column).
- `result.summary()`: one dict row of the run's overall metrics, the shape a study concatenates across
  conditions. Coordinate with study-layer: if a study layer is adopted, this is its row.

## Tests

- Correlation against `numpy.corrcoef` and a hand-worked case; its MC error against the SD over seeded
  replications.
- Length spread against `examinees()["test_length"]` computed by hand.
