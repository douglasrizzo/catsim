---
title: Constant-pattern rules for ML, and a hybrid estimator
kind: technique
status: needs-research
priority: 1
effort: M
blocked_by: []
raised_by: [Wang & Vispoel (1998) walkthrough, Babcock & Weiss (2012) walkthrough]
---

# Constant-pattern rules for ML, and a hybrid estimator

## Problem

While every response is correct, or every response is incorrect, the likelihood has no maximum.
`MLEstimator` then always takes Dodd's variable step toward the bank's hardest or easiest
difficulty and reports the Fisher SE. That is the only rule: there is no fixed step, and no
fallback to a Bayesian estimate. Older CAT studies commonly used one of those, such as EAP or MAP
until the first mixed pattern, or while ML sits at a bound.

Under the 3PL, ML can also sit at a bound with mixed responses, where the bound is a true maximum of
the likelihood. A hybrid estimator that falls back while the estimate is at a bound covers that case
too.

## Evidence

- **Wang & Vispoel (1998) walkthrough.** At θ = +3, about the good bank's highest b, 10.6% of
  simulees were still all correct after 10 items, with the estimate held at about 2.98. That
  flatters ML's bias at the top of the scale. The agent wrote a 20-line `FixedStepMLEstimator` that
  delegates to `MLEstimator`, and it pickled to workers without trouble.
- **Babcock & Weiss (2012) walkthrough.** Under the 3PL, the MLE of some mixed patterns sits at the
  −4 bound. No item is informative there, so a minimum-information stopper ends the test at once.
  With "SE .315 or minimum information .2", RMSE rose from .366 to .798, and 73 of 1,000 examinees
  were off by more than 1.5. With EAP, the same rule cut the mean length from 18.3 to 14.9 at equal
  RMSE. The choice of estimator alone reverses the paper's conclusion.
- catR's `randomCAT` takes a `constantPatt` option. Its believed values are `"BM"`, `"EAP"`,
  `"WL"`, `"fixed4"`, `"fixed7"` and `"var"`; `"var"` reproduces catsim's rule. Verify these
  against catR's source.

## What catsim has now

Dodd's step, documented on the ML technique page and in ADR-0018.
`MAPEstimator(uniform_log_prior(-4, 4), bounds=(-4, 4))` reproduces catR's ML default for constant
patterns.

## Proposal (to settle in research)

- A `constant_patterns=` option on `MLEstimator`: Dodd's step (the default), a fixed
  step, or a fallback estimator. Alternatively, a separate `HybridEstimator(primary, fallback,
  when=...)` that composes two estimators. Composition keeps each estimator single-purpose. Check
  that the result pickles (ADR-0015).
- Decide whether "at a bound" is part of the rule. It needs the bounds, which the estimator knows.
- Write the technique sections of [`_template.md`](../_template.md) into this card, with sources
  (Dodd 1990 for the step rules; the hybrid's source to find), before implementing.

## Tests

- A fixed step of 0.7 from θ = 0 after one correct response gives 0.7: a hand-worked value.
- The hybrid gives the fallback's estimate while the pattern is constant, and the primary's from the
  first mixed pattern.
- A parity case against catR for each `constantPatt` value catsim adopts (ADR-0020).
