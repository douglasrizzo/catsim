**Status:** Implemented
**Module:** {py:class}`catsim.selection.LinearSelector`

# Linear Selector

## Specification

`LinearSelector` administers the items of the list it is given, in the order of
that list. At each step it returns the first item of the list that has not
already been administered.

This selector is useful for deterministic baselines and for reproducing
fixed-form tests through the CAT engine contract. It does not use the current
ability estimate, response vector, or item information.

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. It never returns an already administered item.
2. It follows the order of its list, whatever the order of the items in the bank.
3. Once every item of its list has been administered, it raises `NoItemsAvailableError`, even if other items of the bank remain.

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.LinearSelector
   :members:
   :show-inheritance:
```
