# ADR-0013: Progressive and Proportional Follow the Published Schedule

## Status

Accepted 2026-11-23

## Context

`ProgressiveSelector` and `ProportionalSelector` carry the names of published methods: the
progressive method of Revuelta and Ponsoda (1998) and the proportional method of Segall (2004), in
the form Barrada et al. (2008) gave them, which catR implements. catsim used its own
parameterisation of both:

- The weight on information at position `n` of a test of planned length `N` was
  `min(1, (n - 1) / (N - 1)) ** acceleration`, with `acceleration` positive and 1 (linear) by default.
  Barrada et al.'s schedule is a cumulative power sum with an acceleration parameter `t`. The two
  agree only when both are linear, so `acceleration` was not their `t`.
- `ProportionalSelector` raised the information to `sharpness` times that weight, with
  `sharpness = 6` by default. In the published method the exponent grows to the test length.

The documentation said so, but under the same method names users comparing with catR got different
behaviour, and the default exponent left the end of a test far from maximum-information selection.
At the last item of a 20-item test at θ = 0, over 20 simulated 500-item 3PL banks
(`ItemBank.generate(500, model="3PL", rng=s)` for `s` from 0 to 19), the most
informative item was drawn with probability 0.08 to 0.24 (median 0.12), against 0.32 to 0.96
(median 0.64) with the published exponent of 20.

## Decision

Both selectors use Barrada et al.'s (2008, Equations 12 and 13) schedule, through one shared
helper. The weight at position `n` is the sum of `m ** acceleration` for `m` below `n`, divided by
the same sum for `m` below `N`; `ProgressiveSelector` blends with that weight, and
`ProportionalSelector` raises the information to `N` times it. `acceleration` is Barrada et al.'s
acceleration parameter (`t` for the progressive weight, `s` for the proportional exponent): any
finite number, 0 by default. At 0 the schedule is linear (p. 498) and differs little from Revuelta
and Ponsoda's original one (p. 500).

catsim departs from catR where a session leaves the planned test: past position `N` the weight
stays 1 and the exponent stays at `N` rather than growing, and a one-item test is at its last
position, with weight 1, from the start. Its default acceleration is 0, where catR's `AP` defaults
to 1. The technique pages list these departures.

## Consequences

- Within the planned length, the weights and exponents are those of catR's `nextItem` with
  `criterion = "progressive"` or `"proportional"`, `rule = "length"` and `AP` equal to
  `acceleration`, so results can be compared with catR's.
- At the last item of a planned test, `ProportionalSelector` selects close to maximum information,
  as the published method intends.
- `acceleration` accepts 0 and negative values; only NaN and infinite values are rejected.
- The two selectors share one schedule, so a fix to it reaches both.

## Alternatives Considered

- Keeping catsim's own parameterisation under the published names, documented: users comparing
  with catR would still get different behaviour from methods of the same name.
- Keeping catsim's own schedule behind a parameter: two schedules to test and document for no
  use the published one does not serve.
- Keeping `sharpness`, with `None` meaning the test length: neither Barrada et al. nor catR has such
  a parameter, the published end point already scales with the test, and a second way to spell the
  default would add an optional value and a property that returns either `None` or a number.
