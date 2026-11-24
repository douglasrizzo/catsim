# Randomesque Selector

:Class: {py:class}`catsim.selection.RandomesqueSelector`
:Sources: Kingsbury & Zara (1989)

## Motivation

Maximum-information selection gives examinees of similar ability the same items, so the most
informative items of a bank are overexposed. The randomesque technique of Kingsbury and Zara
(1989) relaxes maximum-information selection by adding controlled randomness among the most
informative items. Magis and Raîche (2012, p. 12) list it among the exposure-control methods that
choose at random among several optimal items near the current estimate.

## Definition

At each step the selector ranks the items that are neither administered nor excluded by Fisher
information at the current ability estimate, keeps the `bin_size` most informative of them, and
draws one uniformly at random from that candidate set. Items tied in information with the last of
those candidates also join the set, as in catR ({doc}`/adr/0011-random-tie-breaking`).

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Parameter
     - Default
     - Description
   * - ``bin_size``
     - required
     - Number of most informative remaining items each item is drawn from, at
       least 1. A bin of 1 is maximum-information selection.
```

## Behavior and Comparison

Unlike the {doc}`5-4-3-2-1 technique <the-54321-selector>`, it keeps drawing from the same
number of best items for the whole test rather than switch to maximum information (Georgiadou et
al., 2007, p. 9). The random draw from the candidate set is its only exposure control.

## Guarantees

1. It never returns an item that was already administered or excluded.
2. It returns one of the `bin_size` most informative remaining items at the current estimate, or an
   item tied with the last of them, drawn with the session's random generator.
3. When no item remains, it raises `NoItemsAvailableError`.
4. It applies no exposure cap: the run's exposure rates do not change its choice.

## References

Kingsbury, G. G., & Zara, A. R. (1989). Procedures for selecting items for computerized adaptive tests. *Applied Measurement in Education*, 2(4), 359-375. <https://doi.org/10.1207/s15324818ame0204_6>. Not openly available; Georgiadou et al. (2007) and Magis and Raîche (2012) describe the method.

Georgiadou, E. G., Triantafillou, E., & Economides, A. A. (2007). A review of item exposure control strategies for computerized adaptive testing developed from 1983 to 2005. *The Journal of Technology, Learning, and Assessment*, 5(8). <https://ejournals.bc.edu/index.php/jtla/article/view/1647> (open access).

Magis, D., & Raîche, G. (2012). Random generation of response patterns under computerized adaptive testing with the R package catR. *Journal of Statistical Software*, 48(8), 1-31. <https://doi.org/10.18637/jss.v048.i08> (open access).

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.RandomesqueSelector
   :members:
   :show-inheritance:
```
