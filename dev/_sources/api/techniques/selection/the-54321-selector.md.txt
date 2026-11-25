# 5-4-3-2-1 Selector

:Class: {py:class}`catsim.selection.The54321Selector`
:Sources: {cite:t}`McBride83`

## Motivation

Under maximum-information selection every examinee starts from the same
ability estimate, so the opening items of every test are the same few items.
The 5-4-3-2-1 technique of {cite:t}`McBride83` randomizes only those
opening items.

## Definition

The opening items are drawn from bins that shrink by one item at each step:

- the first item is drawn uniformly at random from the 5 most informative items
  at the current estimate;
- the second from the 4 most informative remaining items;
- the third from 3 and the fourth from 2;
- from the fifth item on, the most informative remaining item is administered.

This is the schedule described by {cite:t}`Georgiadou2007{p. 9}` in their
review of exposure-control strategies.

With `initial_bin_size` $s$, the item at 0-indexed test position $k$ is drawn
from the $\max(1, s - k)$ most informative items that are neither administered
nor excluded, plus any item tied in information with the last of them
({doc}`/adr/0011-random-tie-breaking`).

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``initial_bin_size``
     - ``5``
     - Number of most informative items the first item is drawn from; each
       following item draws from one fewer, down to 1.
```

## Behavior and Comparison

Unlike the randomesque selector, which keeps drawing from a fixed number of
items for the whole test {cite:p}`Georgiadou2007{p. 9}`, the late items, which
matter most for precision, are chosen by maximum information.
{cite:t}`Magis12{p. 12}` list the technique among the methods that choose at random among
several near-optimal items, without defining it.

## Guarantees

1. It never returns an item that was already administered or excluded.
2. The item at position $k$ is drawn with the session's random generator from the
   $\max(1, s - k)$ most informative remaining items.
3. From position $s - 1$ on, it returns the most informative remaining item.
4. When no item remains, it raises `NoItemsAvailableError`.
5. It applies no exposure cap: the run's exposure rates do not change its choice.

## References

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.The54321Selector
   :members:
   :show-inheritance:
```
