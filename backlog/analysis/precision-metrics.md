---
title: Empirical standard error, robust reported-SE summaries and interval coverage
kind: feature
status: ready
priority: 1
effort: M
blocked_by: []
raised_by: [Wang & Vispoel (1998) walkthrough, Babcock & Weiss (2012) walkthrough]
---

# Empirical standard error, robust reported-SE summaries and interval coverage

Found by the Wang & Vispoel (1998) and Babcock & Weiss (2012) replications of the research walkthrough ([discovery](../discovery/README.md)).

## Problem

1. **No empirical SE.** The central outcome of estimator studies is the conditional SE, the standard
   deviation of θ̂ over replications at one θ. `catsim.metrics`, `SimulationResult` and its tables have
   bias, MSE and RMSE, but no SD of the estimates.
2. **`se` reads as that SE and is not.** `SimulationResult.se()` and the `se` column of `deciles()` and
   `ability_levels()` are the *mean reported* SE. A reader comparing with the literature can take one
   for the other.
3. **The mean reported SE breaks under the 3PL.** ML estimates for some mixed patterns sit at the bound
   (a true maximum of the likelihood under guessing), where the Fisher SE is 10² to 5×10⁶. The mean is
   then useless, and `plot.metric_by_ability(result, "se")` cannot be read.
4. **No check of the reported SE.** Studies report how well the estimator's SE matches the empirical
   SD, and the coverage of θ̂ ± 1.96·SE. Both agents computed these by hand.

## Evidence

- Wang & Vispoel: ML on bounds ±6, θ = −2, 30 items: `ability_levels()["se"]` ≈ 3,270. In the "good"
  pool at θ = −3 after 10 items, 8.2% of ML estimates sit at −6 with mixed responses. Median reported
  SE / empirical SD for ML is 0.30 at θ = −3 after 10 items. EAP's 95% intervals cover only 0.56 at
  θ = −3 after 10 items. Workaround matched `sqrt((mse - bias**2) * N/(N-1))` to 3e-16.
- Babcock & Weiss: MLE on bounds ±4, the mean SE was 16.7 in runs whose typical SE is 0.3; Fisher SE at
  θ̂ = −4 is about 480.

Tables: the walkthrough's `tables/conditional_tables.md` (kept locally),
the walkthrough's `out/tables.md` (kept locally).

## Proposed shape (for the implementer to settle)

- `metrics.sd(thetas)` or `metrics.empirical_se(true_thetas, thetas)` (name by
  `sphinx/terms-and-notation.md`; add the term there if it is missing) with its Monte Carlo error: the
  SD's MC error is approximately `sd / sqrt(2(N-1))` under normality; prefer a form that does not
  assume normality, and cite it.
- A matching `SimulationResult` method and a column in `deciles()` and `ability_levels()`.
- Reported-SE diagnostics: median reported SE, the ratio of mean (or median) reported SE to empirical
  SD, and coverage of θ̂ ± z·SE at a chosen level, each with an MC error where one exists.
- Consider renaming `se` in the tables to say it is the mean reported SE (no aliases before 1.0,
  AGENTS.md). Check the plots that read `se`.

## Tests

- Empirical SE against a hand-worked set of estimates and against `numpy.std(ddof=1)`.
- Its MC error against a brute-force reference: the SD of the SD over many seeded replications.
- Coverage on a hand-built case where the share inside the interval is known.
- A table test that the `se` column and the new SD column differ on a 3PL ML run with bound estimates
  (the bug this card fixes is mistaking one for the other).

## When to start

Any time; independent of the other cards. derived-results's by-length tables would reuse these columns.
