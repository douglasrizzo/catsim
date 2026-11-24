# Selection Techniques

Item selection techniques determine which item from the bank to administer at each
step of an adaptive test. The choice of selector affects both measurement precision
and practical properties such as item exposure and content coverage.

A selector reads the session from a {py:class}`~catsim.context.StepContext`: the
administered items and their responses, the current estimate `theta`, the run's random
generator and exposure rates, and `excluded_items`, items it must pass over although they
have not been administered, such as those the engine's exposure control
({py:class}`~catsim.exposure.BaseExposureControl`) declined. Where the pages below speak of non-administered items, they
mean the items that are neither administered nor excluded. Excluded items do not count
toward the test position, the number of administered items.

## Implemented

```{toctree}
:maxdepth: 1

linear-selector
random-selector
randomesque-selector
the-54321-selector
max-info-selector
urry-selector
interval-info-selector
kl-selector
progressive-selector
proportional-selector
cluster-selector
stratified-selector
a-strat-selector
a-strat-b-block-selector
max-info-strat-selector
max-info-b-block-selector
mei-selector
mlwi-selector
```

## Ties

When several candidate items share the best score exactly, every selector that picks by a score
breaks the tie the same way: it draws one of them uniformly at random with the session's random
generator, as catR does. `ClusterSelector` applies the same rule to its choice of cluster.
Ties are exact equality of the selector's own score, so items that are equally good in exact
arithmetic tie only if their scores also round to the same value.
The Randomesque and 5-4-3-2-1 bins also take in every item tied with their least informative item.
No random number is drawn when nothing ties, so seeded results on banks without ties do not depend
on this rule. See {doc}`/adr/0008-random-tie-breaking`.

## Class Hierarchy

```{autoclasstree} catsim.selection
:direction: LR
:full:
:strict:
:namespace: catsim
```
