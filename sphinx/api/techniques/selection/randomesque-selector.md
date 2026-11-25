# Randomesque Selector

:Class: {py:class}`catsim.selection.RandomesqueSelector`
:Sources: {cite:t}`Kingsbury89`

## Motivation

Maximum-information selection gives examinees of similar ability the same items, so the most
informative items of a bank are overexposed. The randomesque technique of {cite:t}`Kingsbury89`
relaxes maximum-information selection by adding controlled randomness among the most
informative items. {cite:t}`Magis12{p. 12}` list it among the exposure-control methods that
choose at random among several optimal items near the current estimate.

## Definition

At each step the selector ranks the items that are neither administered nor excluded by Fisher
information at the current ability estimate, keeps the `bin_size` most informative of them, and
draws one uniformly at random from that candidate set. Items tied in information with the last of
those candidates also join the set, as in catR ({doc}`/adr/0017-random-tie-breaking`).

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
number of best items for the whole test rather than switch to maximum information
{cite:p}`Georgiadou2007{p. 9}`. The random draw from the candidate set is its only exposure control.

## Guarantees

1. It never returns an item that was already administered or excluded.
2. It returns one of the `bin_size` most informative remaining items at the current estimate, or an
   item tied with the last of them, drawn with the session's random generator.
3. When no item remains, it raises `NoItemsAvailableError`.
4. It applies no exposure cap: the run's exposure rates do not change its choice.

## References

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.RandomesqueSelector
   :members:
   :show-inheritance:
```
