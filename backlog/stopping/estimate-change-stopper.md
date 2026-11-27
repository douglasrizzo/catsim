---
title: Estimate-change stopper, the previous estimate in StepContext, and notebook 04's recipe
kind: fix
status: ready
priority: 1
effort: M
blocked_by: []
raised_by: [Babcock & Weiss (2012) walkthrough]
---

# Estimate-change stopper, the previous estimate in StepContext, and notebook 04's recipe

Found by the Babcock & Weiss (2012) replication of the research walkthrough ([discovery](../discovery/README.md)).

## Problem

Stopping when the estimate changes by less than Δ is a standard termination criterion (Babcock & Weiss
2012), and catsim has no built-in for it. Notebook 04 shows how to write one, `EstimateChangeStopper`:
it re-estimates the previous estimate by passing the estimator the context without the newest item,
built with `dataclasses.replace`. That context still holds the newest estimate as `theta`.

With `NumericalSearchEstimator`, while every response is correct or every response is incorrect, the
estimate is Dodd's step from `context.theta`. The re-estimate then steps from the *new* estimate, not
the one before it, so the stopper sees the wrong change. The notebook's warning covers only estimators
that "start their search" from the current estimate. Dodd's step is not a search, so a reader using ML
gets silently wrong stops.

## Evidence

Measured against the estimate history recorded in `session.estimates`
(the walkthrough's `check_notebook_stopper.py` (kept locally)):

- 38 of 1,000 examinees on the peaked 500-item bank stopped at the wrong item.
- 3 of 1,000 on the flat 100-item bank stopped at the wrong item.

The agent's workaround replays the chain of Dodd steps from θ0 and needs its own copy of the engine's
estimator:

```python
if len(set(responses)) > 1:
    return est.estimate(replace(ctx, administered_items=items, responses=responses)).theta
theta = self.theta0
for k in range(1, len(items) + 1):
    theta = est.estimate(replace(ctx, administered_items=items[:k], responses=responses[:k], theta=theta)).theta
```

The same study found Δθ the weakest rule, as the paper did. At the extremes of the peaked banks it
stops after about 6 items with bias −1.03 at θ = 3, because Dodd's geometric steps look like
convergence. That is the method's behaviour, not a bug; the technique page should say so.

## What to do

1. **Now (docs):** correct the notebook's explanation. The recipe is exact only for an estimator that
   ignores `context.theta`. `NumericalSearchEstimator` uses it for Dodd's step, so under ML the recipe
   is wrong while all responses are the same. Point to the built-in once it exists. Run the notebook
   with nbmake.
2. **The real fix (architecture):** a stopper needs the estimate before the newest response. Options:
   - Add the session's estimate history, or only the previous estimate, to `StepContext`. This changes
     ADR-0007's strategy contract, so it needs an ADR update. ADR-0006 says catsim keeps nothing
     between calls, and the history is already in the caller's `Session`, so passing it through the
     context keeps that rule.
   - A built-in `EstimateChangeStopper` that reads it.
   - Do not have the stopper hold state (ADR-0007: strategies hold no session data).
3. **Spec:** add the technique sections of [`_template.md`](../_template.md) to this card first,
   with Babcock & Weiss (2012) as the source.

## Tests

- The built-in stopper's decisions on a seeded ML run equal those computed from `session.estimates`,
  session by session. This is the bug the notebook recipe had.
- A hand-worked context where the change is just below and just above Δ.

## Related

- The minimum-information stopper the same study needed already has a spec:
  [min-info-bank-stopper](min-info-bank-stopper.md).
