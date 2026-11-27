---
title: Exposure-evenness summaries (χ² index, unused and over-exposed shares)
kind: feature
status: ready
priority: 2
effort: S
blocked_by: []
raised_by: [Chang & Ying (1999) walkthrough]
---

# Exposure-evenness summaries (χ² index, unused and over-exposed shares)

Found by the Chang & Ying (1999) replication of the research walkthrough ([discovery](../discovery/README.md)).

## Problem

Every exposure-control study reports how evenly the bank is used: the χ² index of Chang & Ying (1999),
the share of items never administered, the share above the cap r, and a table of items per exposure
band. catsim has `exposure_rates()` and `overlap_rate()` only, so the agent computed the rest by hand.

## Evidence

The agent's main table (the walkthrough's `tables.md` (kept locally)), means over 5
replications of 5,000 examinees, 360 items, L = 40:

| | max r | % unused | % > .20 | χ² | overlap |
|---|---|---|---|---|---|
| MI | .81 | 51.2 | 22.7 | 112.3 | .423 |
| a-STR | .38 | 0.3 | 9.7 | 14.3 | .151 |

The agent checked χ² against `overlap_rate()` through the identity χ² = M·T − L (M items, T the overlap
rate, L the test length).

## Proposed shape (for the implementer to settle)

- `metrics.exposure_chi_square(rates, test_size)`: Σ (rᵢ − L/M)² / (L/M). Check the definition and its
  source page in Chang & Ying (1999) and Chen, Ankenmann & Spray (2003) before implementing; document
  the identity with the overlap rate on the technique page, not in the docstring.
- A `SimulationResult` method or table giving χ², the share unused, the share above a given r and
  counts per band; fixed-length tests only where the formula needs L, like `overlap_rate`.
- Consider whether the bands belong in catsim or in the caller's `numpy.histogram`; keep only what is
  more than a one-liner (YAGNI).

## Tests

- χ² on a hand-worked bank of a few items.
- The identity χ² = M·T − L against `metrics.overlap_rate` on a seeded run (an independent route to the
  same number).
