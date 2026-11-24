# ADR-0012: Scope 1.0 to Unidimensional, Dichotomous IRT

## Status

Accepted

## Context

Every contract in catsim assumes one ability and right-or-wrong answers. An ability estimate is a
`float`, a response is a `bool`, and an item bank has the four columns of the logistic models, `a`,
`b`, `c` and `d`. `StepContext`, `CatSessionState` and the functions in `catsim.irt` are typed and
validated that way.

Multidimensional IRT (MIRT) would replace the ability with a vector and each discrimination with
one per dimension, which reaches `catsim.irt`, `ItemBank`, `CatSessionState` and every initializer,
selector, estimator and stopper. Polytomous items, such as partial-credit or graded-response items,
would replace the Boolean responses and the item parameters. Neither is planned, but nothing stated
that 1.0 leaves them out, and the contracts frozen at 1.0 would decide how either could be added.

## Decision

The 1.0 contracts cover unidimensional IRT with dichotomous items: the one- to four-parameter
logistic models. A scalar ability estimate, Boolean responses and the four-column item bank are part
of the stable API.

Support for multidimensional abilities or polytomous items would come as a new major version, or as
parallel types beside the current ones, not by widening the current types to a scalar-or-vector
ability or a Boolean-or-category response.

## Consequences

- Strategies can rely on the ability being one `float` and the responses being Booleans, without
  checks for other shapes.
- Planned techniques that stay within the scope, such as the classification stoppers and the
  exposure controls, can be added without breaking the contracts.
- MIRT or polytomous support is a breaking change, or a separate set of types, and gets its own
  design when it is planned.
