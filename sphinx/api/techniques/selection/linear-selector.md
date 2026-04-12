**Status:** Implemented
**Module:** {py:class}`catsim.selection.LinearSelector`

# Linear Selector

## Specification

`LinearSelector` administers items in their fixed bank order. At each step it
returns the first item in that order that has not already been administered.

This selector is useful for deterministic baselines and for reproducing
fixed-form tests through the CAT engine contract. It does not use the current
ability estimate, response vector, or item information.

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. It never returns an already administered item.
2. It preserves the item bank order when choosing the next item.
3. If no non-administered items remain, it raises `NoItemsAvailableError`.

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.LinearSelector
   :members:
   :show-inheritance:
```
