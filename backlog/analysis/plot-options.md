---
title: Comparable exposure histograms and more metrics in the plots
kind: feature
status: ready
priority: 3
effort: S
blocked_by: []
raised_by: [Chang & Ying (1999) walkthrough, Babcock & Weiss (2012) walkthrough]
---

# Comparable exposure histograms and more metrics in the plots

Found by the Chang & Ying (1999) and Babcock & Weiss (2012) replications of the research walkthrough ([discovery](../discovery/README.md)).

## Problem

1. `plot.item_exposure(..., hist=True)` takes no `bins` or `range`, so each method's panel gets its own
   bin edges and panels cannot be compared side by side
   (the walkthrough's `figures/fig5_exposure_hist.png` (kept locally)).
2. No exposure plot can draw the L/M reference line (the rate every item has under perfectly even use);
   the agent added it by hand.
3. `plot.metric_by_ability` cannot plot the correlation or the length SD (it plots the metrics of
   `ability_levels()`/`deciles()`, so it follows precision-metrics and run-summary).

## Proposed shape

- `bins` (passed to matplotlib) on `item_exposure`'s histogram view. Check whether a `range` argument
  is needed once bins can be an array of edges; one argument is better than two.
- A reference line: either an argument, or a documented `ax.axhline(L / M)` in the guide. Prefer the
  guide unless several plots need it.
- `metric_by_ability`'s supported metrics follow the table columns precision-metrics and run-summary add.

## Tests

- Two `item_exposure` histograms with the same `bins` share their edges (read back from the axes'
  patches).
