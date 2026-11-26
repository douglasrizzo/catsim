# ADR-0002: Unidimensional, Dichotomous IRT Only

## Status

Accepted 2026-11-25

## Context

Every contract in catsim assumes one ability and right-or-wrong answers. An ability estimate is a
`float`, a response is a `bool`, and an item bank has the four columns of the logistic models, `a`,
`b`, `c` and `d`. The step context, the session and the functions in `catsim.irt` are typed and
validated that way.

Multidimensional IRT (MIRT) would replace the ability with a vector and each discrimination with
one per dimension, which reaches `catsim.irt`, the item bank, the session and every strategy.
Polytomous items, such as partial-credit or graded-response items, would replace the Boolean
responses and the item parameters.

## Decision

catsim covers unidimensional IRT with dichotomous items: the one- to four-parameter logistic models.
A scalar ability estimate, Boolean responses and the four-column item bank are part of its
contracts throughout 1.x.

Multidimensional abilities or polytomous items would come in a later major version, with their own
design. The 1.x design does not anticipate them: no type is widened, and no abstraction is added,
for models catsim does not implement.

## Consequences

- Strategies rely on the ability being one `float` and the responses being Booleans, without checks
  for other shapes.
- Techniques within the scope, such as the classification stoppers and the exposure controls, can
  be added without breaking the contracts.
- Researchers who need polytomous items or MIRT use other software.

## Alternatives Considered

- Widening the types now, to a scalar-or-vector ability or a Boolean-or-category response: every
  strategy would check which form it received, for models no part of catsim implements.
- Designing extension points for other models now: abstractions for needs nobody has yet, which a
  real design for those models would likely replace.
- Leaving the scope unstated: a proposal for MIRT or polytomous items would find no record of what
  may change and how.
