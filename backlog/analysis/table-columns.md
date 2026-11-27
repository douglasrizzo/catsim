---
title: Constant-pattern, bound and stop-criterion columns in the result tables
kind: feature
status: ready
priority: 2
effort: S
blocked_by: []
raised_by: [Wang & Vispoel (1998) walkthrough, Babcock & Weiss (2012) walkthrough]
---

# Constant-pattern, bound and stop-criterion columns in the result tables

Found by the Wang & Vispoel (1998) and Babcock & Weiss (2012) replications of the research walkthrough ([discovery](../discovery/README.md)).

## Problem

1. **Constant patterns and bounds.** Estimator studies report how often responses are still all
   correct or all incorrect, and how often ML sits at a bound. The agent recomputed both from the
   `response` column and `abs(theta) >= 5.999`.
2. **Which criterion stopped the test.** With `StopRule(StandardErrorStopper(...), MinInfoStopper(...))`,
   a stopping-rule study reports how many tests each criterion ended. `examinees()["stop_reason"]` is
   only `"stopper"` or `"no_items"`. The agent looped `rule.satisfied(engine.context(session))` over the
   sessions (0.03 s per 1,000).

## Evidence

- Wang & Vispoel, good pool, θ = −3 after 10 items: 8.2% of ML estimates at −6 with mixed responses;
  on bounds ±4, 8.8% at −4. At θ = +3, 10.6% still all correct after 10 items
  (the walkthrough's `tables/constant_patterns.md` (kept locally)).
- Babcock & Weiss, peaked 500-item bank, SE .315 or MI .2: 934 tests ended by SE and 66 by MI.

## Proposed shape (for the implementer to settle)

- `steps()` and `examinees()`: a `constant_pattern` column (responses so far all equal).
- An at-bound flag needs the estimator's bounds, which a result does not know; options: the engine
  passes them, the `Estimate` carries a flag, or catsim leaves it to the caller. `Estimate` is in
  ADR-0007's strategy contract, so a new field there needs that ADR updated. Prefer the smallest
  option; drop the flag if none is clean.
- Stop criterion: the result does not hold the engine, so this is either a column `simulate` fills
  from `StopRule.satisfied` at the last step (the session would need to record it: check ADR-0010's
  `StopReason`), or a documented recipe with `rule.satisfied(engine.context(session))`. Choose with the
  ADR in hand.

## Tests

- `constant_pattern` against a hand-built session.
- The stop criterion on a seeded run where each criterion ends some tests, checked against
  `rule.satisfied` per session.
