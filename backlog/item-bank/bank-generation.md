---
title: Generate banks of a chosen shape, and document the scaling constant D where users look
kind: decision
status: needs-decision
priority: 2
effort: S
blocked_by: []
raised_by: [Wang & Vispoel (1998) walkthrough, Babcock & Weiss (2012) walkthrough, Chang & Ying (1999) walkthrough]
---

# Generate banks of a chosen shape, and document the scaling constant D where users look

Found by all three replications of the research walkthrough ([the research walkthrough](../discovery/README.md)).

## Problem

1. **Fixed distributions.** `ItemBank.generate(n_items, model, corr, rng)` draws a ~ N(1.2, 0.25)
   truncated, b ~ N(0, 1), c ~ N(0.25, 0.02) and d ~ U(0.94, 1). Simulation studies vary the bank's
   shape, so all three agents built their item matrices with NumPy:
   - Wang & Vispoel: a "good" bank with b ~ U(−3, 3) and a "poor" bank.
   - Babcock & Weiss: flat banks with b ~ U(−3.5, 3.5) and peaked banks with b ~ N(0, 0.75), with a
     fixed c = 0.2.
   - Chang & Ying: a ~ lognormal, because the default a range of 0.40 to 1.86 is narrow for a study
     that stratifies by a.
   Building the matrix was easy in every case (`ItemBank(numpy.column_stack([a, b, c, d]))`). The gap
   is that nothing shows how.
2. **D = 1.7.** catsim's logistic functions have no scaling constant, so parameters calibrated with
   D = 1.7 must have a multiplied by 1.7. This is said only in notebook 01 and on the stratified
   selector page, not in the `ItemBank` or `catsim.irt` docstrings, where a user builds a bank.

## What to do

- Pick one:
  - Let `generate` take the distributions, such as a callable or frozen SciPy distribution per
    parameter.
  - Document a recipe for building the matrix yourself, in the guide that covers banks.

  The recipe is the smaller change and may be enough, since all three agents managed alone (YAGNI).
  Decide with the maintainer.
- State the absent D in the `ItemBank` constructor's docstring and in `catsim.irt`'s module docstring,
  as part of what the parameters mean. Give the rescaling once, in the guide or the terms page, and
  link to it from the others (one home per topic).
- The alternative is a `D` argument: on `ItemBank`, which would rescale a once, or on every IRT
  function and strategy, as catR has. Prefer the documentation: the item matrix stays the one thing
  that defines an item (ADR-0003), and every caller would otherwise have to pass the same D.

## Tests

- If `generate` takes distributions: a seeded bank drawn from given distributions has the parameters
  those distributions give for the same generator stream.
