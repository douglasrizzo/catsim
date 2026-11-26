# Linear Selector

:Class: {py:class}`catsim.selection.LinearSelector`

## Motivation

`LinearSelector` gives a fixed-form test through the CAT engine: every examinee receives the same
items in the same order. It serves as a deterministic baseline for adaptive selectors, and lets a
fixed form be scored with the same estimators and stoppers as an adaptive test. It does not use
the ability estimate, the responses or the item information.

## Definition

The selector administers the items of the list it is given, in the order of that list. At each
step it returns the first item of the list that is neither administered nor excluded.

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Parameter
     - Default
     - Description
   * - ``indices``
     - required
     - Non-empty list of non-negative item indices, administered in the order
       given.
```

## Behavior and Comparison

With an exposure control, each declined item is lost for the rest of the test,
so the list can run out before the test is complete, and the session then ends
with `StopReason.NO_ITEMS`.

## Guarantees

1. It never returns an item that was already administered or excluded.
2. It follows the order of its list, whatever the order of the items in the bank.
3. Once every item of its list has been administered or excluded, it raises
   `NoItemsAvailableError`, even if other items of the bank remain.
4. If an item of its list is not an index of the item bank, it raises `ValueError` rather than
   return that index.

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.LinearSelector
   :members:
   :show-inheritance:
```
