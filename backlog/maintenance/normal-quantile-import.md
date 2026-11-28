---
title: Compute normal quantiles with scipy.special instead of importing scipy.stats
kind: maintenance
status: ready
priority: 3
effort: XS
blocked_by: []
raised_by: [import-time measurement of 2026-11-26]
---

# Compute normal quantiles with scipy.special instead of importing scipy.stats

## Problem

`import catsim` takes about 0.4 seconds, and nearly half of it is `scipy.stats`, which catsim uses for
one thing: the standard normal quantile. Every script, test run, worker process and package that
imports catsim pays it.

## Evidence

Measured on 2026-11-26 at `97b89f50`, CPython 3.13.6 on an Apple M4 Pro, NumPy 2.5.5 and SciPy
1.18.1, timing each import in turn in a fresh process (three runs, all within 0.01 s of these):

| Import | Seconds |
|---|---|
| `numpy` | 0.03 |
| `scipy.special` | 0.10 |
| `scipy.optimize` | 0.07 |
| `scipy.stats`, after the three above | 0.18 |
| the rest of `catsim` | 0.015 |
| total | 0.40 |

`import catsim` loads `irtlib.irt`, the item response functions catsim builds on (ADR-0024), which
imports `scipy.stats` at module level.

## What catsim has now

Two of catsim's modules import `scipy.stats`, each only to call `stats.norm.ppf((1 + confidence) / 2)`:

- `src/catsim/stopping/precision.py`, in `ConfidenceIntervalStopper`;
- `src/catsim/selection/information.py`, in `IntervalInfoSelector`.

The third import is irtlib's, in `irtlib/irt.py` for `confidence_interval`, which `import catsim`
loads through `irtlib.ItemBank`. irtlib's own card, `irt/normal-quantile-import`, replaces it.

## Proposal

Replace each `stats.norm.ppf(p)` with `scipy.special.ndtri(p)`, the inverse of the standard normal
distribution function, and drop the two `from scipy import stats` lines. `import catsim` stops
loading `scipy.stats` once irtlib's card lands too.

## Tests

- After `import catsim` in a fresh interpreter, `"scipy.stats"` is not in `sys.modules`. The test
  fails on the current code, and passes once irtlib's card has landed too.
- The precision stopper and the selector that use a confidence interval keep their parity cases
  against catR.
