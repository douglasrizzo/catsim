# Random Selector

:Class: {py:class}`catsim.selection.RandomSelector`

## Motivation

`RandomSelector` is a non-adaptive baseline: it ignores the ability estimate, the responses and
the item parameters. Comparing an adaptive selector against it shows how much of a test's
precision comes from choosing items adaptively. Because every item is equally likely, it also
spreads exposure as evenly as a selector can.

## Definition

At each step the selector draws one item uniformly at random from the items that are neither
administered nor excluded.

## Parameters

`RandomSelector` takes no parameters.

## Guarantees

1. It never returns an item that was already administered or excluded.
2. It draws with the session's random generator, so a run with the same seed administers the same
   items.
3. When no item remains, it raises `NoItemsAvailableError`.

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.RandomSelector
   :members:
   :show-inheritance:
```
