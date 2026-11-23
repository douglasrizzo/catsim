# ADR-0008: Break Ties Between Items Uniformly at Random

## Status

Accepted

## Context

Selectors that pick the best item by a score broke ties in three different ways:

- `MaxInfoSelector` and `UrrySelector` took the first item in numpy's default `argsort` order. That
  sort is not stable, so the tied item they chose followed no rule: it changed when unrelated items
  were added to the bank or administered.
- `IntervalInfoSelector`, the stratified selectors, `ClusterSelector` and the newer selectors took
  the lowest index through `argmax` or `argmin`.
- The Randomesque and 5-4-3-2-1 bins cut a sorted list at a fixed size, so which tied item made the
  bin also followed the unstable sort.

Ties are common: duplicated items, 1PL banks with repeated difficulties, and items whose
difficulties sit symmetrically around the ability estimate. Mathematically tied items are also
tied bit for bit, so the choice among them is a real behaviour, not a rounding accident. A fixed
order also hands all the exposure of a set of duplicates to one copy until it is administered.

catR breaks every tie the same way: it keeps the items whose score equals the best score exactly
and draws one of them with `sample()`. For randomesque bins, every item scoring at least the
cutoff joins the pool.

## Decision

Every selector that picks the best item by a score uses one rule, catR's: the candidates whose
score equals the best score exactly are tied, and one of them is drawn uniformly with the session's
random generator. Randomesque and 5-4-3-2-1 bins hold every candidate tied with the bin's least
informative item. `BaseSelector._pick_best` and `BaseSelector._best_pool` implement the rule.

The rule is fixed. There is no parameter, enum or package setting to change it.

When there is no tie, no random number is drawn. When items tie and no generator was passed, the
selector raises `ValueError`, as the random selectors already do without a generator.
`CatEngine` always passes the session generator.

`ClusterSelector` applies the rule to its choice of item within a cluster, but not yet to its
choice of cluster.

## Consequences

- Seeded results change only on banks with exact ties. On untied banks, picks and the random
  number stream are the same as before.
- Exposure is shared evenly among duplicated items.
- Callers that use a deterministic selector directly, without the engine, must pass `rng` when
  their bank can hold ties.
- `MaxInfoSelector` uses the shared exposure filter, which also makes it 15 to 50 times faster
  on banks of 500 to 50,000 items.
