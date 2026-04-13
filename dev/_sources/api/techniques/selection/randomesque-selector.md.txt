**Status:** Implemented
**Module:** {py:class}`catsim.selection.RandomesqueSelector`

# Randomesque Selector

## Specification

`RandomesqueSelector` ranks non-administered items by Fisher information at
the current ability estimate, keeps the top `bin_size` candidates, and then
selects uniformly at random from that candidate set.

The technique relaxes maximum-information selection by adding controlled
randomness among the most informative items.

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. It considers only non-administered items.
2. It builds its random candidate set from the most informative remaining items.
3. It samples from the candidate set using the configured random number generator.
4. If no non-administered items remain, it raises `NoItemsAvailableError`.

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.RandomesqueSelector
   :members:
   :show-inheritance:
```
