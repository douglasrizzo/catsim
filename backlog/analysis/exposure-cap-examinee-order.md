---
title: Say that examinee order changes a run with an exposure cap
kind: docs
status: ready
priority: 1
effort: XS
blocked_by: []
raised_by: [Chang & Ying (1999) walkthrough]
---

# Say that examinee order changes a run with an exposure cap

Found by the Chang & Ying (1999) replication of the research walkthrough ([discovery](../discovery/README.md)).

## Problem

`simulate` tests the examinees in the order of `true_thetas`. With exposure feedback, such as
`MaxInfoSelector(r_max=...)`, each session sees the exposure of the sessions before it, so the order
changes the results. A fixed-θ grid built the obvious way, `numpy.repeat(grid, 500)`, puts all 500
examinees at θ = −3 first, and the cap then binds on the easy items among them alone.

`SimulationResult.ability_levels()` documents exactly that design ("100 at each of 13 abilities from
-3 to 3"), and nothing warns that a sorted grid skews a capped run.

## Evidence

360-item 3PL bank, L = 40, EAP, `MaxInfoSelector(r_max=0.2)`, 13 θ points × 500 examinees. At
θ = −3 (the walkthrough's `extras.log` (kept locally)):

| grid order | MSE | bias |
|---|---|---|
| sorted (`numpy.repeat`) | 0.94 | +0.87 |
| shuffled | 0.37 | +0.50 |

## What catsim says now

`sphinx/running-simulations.md` says that `simulate` "gives one test to each true ability, in the order
of `true_thetas`", and that "A cap counts the exposure of the examinees tested before the current one",
but it never puts the two together. The `r_max` docstrings and `ability_levels()` say nothing about
order.

## Fix

- In `running-simulations.md`, next to the cap paragraph: with exposure feedback the order of
  `true_thetas` is part of the design. Shuffle a grid, such as
  `true_thetas = rng.permutation(numpy.repeat(grid, 500))`; the result keeps the shuffled order, and
  `ability_levels()` groups by ability whatever the order.
- In `simulate`'s docstring (the call contract): the examinees are tested in the order given, and with
  exposure feedback that order changes the result.
- Optional, as a separate decision: a `shuffle` argument to `simulate` that tests in a random order
  drawn from `rng` and still returns the sessions in input order. Only worth it if the doc fix proves
  insufficient (YAGNI).

## Tests

None for the docs. Run the new snippet by hand (no test runs Sphinx code).
