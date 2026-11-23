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
difficulties sit symmetrically around the ability estimate. Duplicates and repeated difficulties
tie bit for bit, so the choice among them is a real behaviour, not a rounding accident. A fixed
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

Ties are exact equality of each selector's own score, as in catR, so they depend on how the score
rounds. Two items whose difficulties sit symmetrically around the estimate tie in information, but
not always in distance: in floating point, |0.3 - 0.1| is 0.19999999999999998 and |-0.1 - 0.1| is
0.2. `UrrySelector` and the stratified selectors, which score by distance, then pick the nearer
item, as catR's `bOpt` does. On 1PL banks, `MaxInfoSelector` and the bins now rank by information,
as catR's `MFI` does, instead of by distance to the estimate. Information is flat near its peak, so
items whose distances differ only by rounding tie there.

`ClusterSelector` applies the rule twice: to its choice of cluster, among clusters whose scores tie
exactly, and to its choice of item within that cluster. Its `item_info` rule scores clusters by
information on every bank, 1PL included, as `MaxInfoSelector` does.

## Consequences

- Seeded results change only where items tie exactly in a selector's score. On banks with
  continuous parameters, picks and the random number stream are the same as before; on 1PL banks
  this excludes items whose difficulties differ only by rounding, which now tie for
  `MaxInfoSelector` and the bins.
- Exposure is shared evenly among duplicated items.
- Callers that use a deterministic selector directly, without the engine, must pass `rng` when
  their bank can hold ties.
- `MaxInfoSelector` uses the shared exposure filter, which also makes it 15 to 50 times faster
  on banks of 500 to 50,000 items.
