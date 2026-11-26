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

## Behavior and Comparison

At a given test length, its tests are less precise than those of any selector that
uses the item parameters, and each item's exposure rate tends to the test length
divided by the number of items in the bank.

:::{dropdown} Comparison with catR

The selector is catR's `nextItem` with `criterion = "random"`, which also draws
uniformly among the available items. R and NumPy draw different numbers, so only
a simulation could compare the two, and the parity tests have none for random
selection ({doc}`ADR-0020 </adr/0020-catr-as-the-reference>`).
:::

## Guarantees

1. It never returns an item that was already administered or excluded.
2. It draws with the generator of the call, so a run with the same seed administers the same
   items.
3. When no item remains, it raises `NoItemsAvailableError`.

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.RandomSelector
   :members:
   :show-inheritance:
```
