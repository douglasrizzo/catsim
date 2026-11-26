# Selection Techniques

```{py:module} catsim.selection
```

Item selection techniques determine which item from the bank to administer at each
step of an adaptive test. The choice of selector affects both measurement precision
and practical properties such as item exposure and content coverage.

A selector reads the session from a {py:class}`~catsim.StepContext`: the
administered items and their responses, the current estimate `theta`, the random generator
of the call, the exposure rates, and `excluded_items`, items it must pass over although they
have not been administered, such as those the engine's exposure control
({py:class}`~catsim.exposure.BaseExposureControl`) declined. Where the pages below speak of remaining items, they
mean the items that are neither administered nor excluded. Excluded items do not count
toward the test position, the number of administered items.

```{toctree}
:caption: Information-based
:maxdepth: 1

max-info-selector
urry-selector
interval-info-selector
kl-selector
mei-selector
mlwi-selector
```

```{toctree}
:caption: Randomized and exposure control
:maxdepth: 1

random-selector
randomesque-selector
the-54321-selector
progressive-selector
proportional-selector
stratified-selector
```

```{toctree}
:caption: Other
:maxdepth: 1

linear-selector
cluster-selector
```

(selection-ties)=

## Ties

When several candidate items share the best score exactly, every selector that picks by a score
breaks the tie the same way: it draws one of them uniformly at random with the generator of the
call, as catR does. `ClusterSelector` applies the same rule to its choice of cluster.
Ties are exact equality of the selector's own score, so items that are equally good in exact
arithmetic tie only if their scores also round to the same value.
The randomesque and 5-4-3-2-1 bins also take in every item tied with their least informative item.
No random number is drawn when nothing ties, so seeded results on banks without ties do not depend
on this rule. See {doc}`/adr/0017-random-tie-breaking`.

(selection-exposure-cap)=

## Exposure cap

Several selectors take an exposure cap, `r_max`. They prefer the remaining items whose exposure rate
is below it, and when every remaining item is at or above it they choose among all the remaining
items, so the cap never ends a test; the cluster selector applies its cap inside the cluster it
chooses, as its page says. The default of 1 applies no cap. The rates are those of the context:
{py:func}`~catsim.simulate` gives each session catR's rates, each item's administrations over the
sessions before it ({doc}`/adr/0012-exposure-rates`), and an application passes its own to
{py:meth}`CatEngine.select <catsim.CatEngine.select>`. A selector that takes no cap says so on its
page.

catR caps exposure only in `simulateRespondents`, with `rmax`. Under its default rule,
`Mrmax = "restricted"`, every item at or above the cap is unavailable to the next examinee, with no
fallback; under `Mrmax = "IE"`, the item-eligibility method, each item is available with a
probability that the run updates from the exposure rates, a method catsim does not implement. No
parity test covers the cap ({doc}`/adr/0020-catr-as-the-reference`).

## Class Hierarchy

```{autoclasstree} catsim.selection
:direction: LR
:full:
:strict:
:namespace: catsim
```

## Base Class and Helpers

Every selector subclasses {class}`~catsim.selection.BaseSelector`. The functions after it are the
rules the built-in selectors share, for custom selectors that should behave alike.

```{eval-rst}
.. autoclass:: catsim.selection.BaseSelector
   :members:

.. autofunction:: catsim.selection.remaining

.. autofunction:: catsim.selection.below_cap

.. autofunction:: catsim.selection.pick_best

.. autofunction:: catsim.selection.best_pool

.. autofunction:: catsim.selection.check_r_max

.. autofunction:: catsim.selection.check_test_size
```
