# ADR-0009: Ability Scale and Extreme Response Patterns

## Status

Accepted

## Context

`catsim.irt` defines two ability ranges: `THETA_MIN_TYPICAL` and `THETA_MAX_TYPICAL` (±4) and
`THETA_MIN_EXTENDED` and `THETA_MAX_EXTENDED` (±6). Every search and grid in the package defaults to
the extended range, but the estimators used it differently:

- `NumericalSearchEstimator` and `WarmLikelihoodEstimator` searched [-6, 6], with no way to change it.
- `MAPEstimator` searched its `bounds` parameter, [-6, 6] by default.
- `EAPEstimator` averaged over its quadrature grid, by default 121 nodes on [-6, 6].

After an all-correct or all-incorrect response pattern the likelihood keeps rising toward one end of
the scale, and the maximum likelihood estimate is infinite. `NumericalSearchEstimator(dodd=False)`
returned that infinity. The engine stored it as the session's estimate, every item then had zero
information at it, and the selectors picked uniformly from the whole bank (ADR-0008); a session that
ended on such a pattern reported an infinite estimate, and `MEISelector` needed a special case for
infinite look-ahead estimates. With `dodd=True`, the default, the estimator took Dodd's step half
the distance to the bank's hardest or easiest item, which moved the estimate down after correct
answers when it was already above the hardest item (4.0 to 3.0), and was not clipped to the search
range.

## Decision

The ability scale's default range is [-6, 6], six standard deviations either side of the mean of a
standard normal population: wider than any plausible examinee or item bank, and narrow enough that
every search ends.

Every estimator returns a finite estimate inside its bounds once an item has been administered,
whatever the response pattern. `BaseEstimator.estimate` states this as the contract, and the engine
and the selectors rely on it. The estimators that search the scale, `NumericalSearchEstimator`,
`MAPEstimator` and `WarmLikelihoodEstimator`, take the same `bounds` parameter, by default
`(THETA_MIN_EXTENDED, THETA_MAX_EXTENDED)`, and reject bounds that are not finite with the lower one
below the upper. `EAPEstimator`'s range is its quadrature grid. With no administered item,
`NumericalSearchEstimator` and `WarmLikelihoodEstimator` return the estimate they were given, and
`EAPEstimator` and `MAPEstimator` the prior's mean and mode.

What each estimator does with an all-correct or all-incorrect pattern:

- `NumericalSearchEstimator` always takes Dodd's step (Dodd, 1990): half the distance from the
  current estimate to the bank's largest difficulty after correct answers, or to its smallest after
  incorrect ones. The step never moves against the responses, so an estimate already beyond that
  difficulty stays where it is, and the result is clipped to the bounds. There is no switch to turn
  it off.
- `EAPEstimator` returns the posterior mean over its grid, which the prior pulls toward the prior
  mean.
- `MAPEstimator` returns the posterior mode inside its bounds. With the default normal prior the mode
  is interior; with `uniform_log_prior()` it is, to within `tol`, the bound the responses point to,
  which is the maximum likelihood estimate restricted to the bounds.
- `WarmLikelihoodEstimator` returns the root of Warm's modified score, which is finite for these
  patterns, or the bound the modified score points to when the root lies beyond the bounds.

`THETA_MIN_TYPICAL` and `THETA_MAX_TYPICAL` do not bound any estimate.

## Consequences

- No estimator returns an infinite estimate, so session histories, simulation results and the
  scores selectors compute are always finite. `MEISelector` no longer special-cases infinite
  look-ahead estimates.
- The `dodd` parameter of `NumericalSearchEstimator` and `catsim.cat.dodd` are removed. Users who
  want the bounded maximum likelihood estimate for constant patterns use
  `MAPEstimator(log_prior=uniform_log_prior())`, which returns -6 or 6 to within its `tol`.
- Seeded results change only where Dodd's step changed: sessions whose estimate starts beyond the
  bank's easiest or hardest item, and banks whose extreme difficulties lie beyond the bounds.
- Narrower `bounds` restrict mixed response patterns too: their estimate is then the bound when the
  likelihood's maximum lies beyond it.
- A custom estimator must keep the contract; nothing downstream checks it.
