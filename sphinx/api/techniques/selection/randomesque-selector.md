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

## References

Kingsbury, G. G., & Zara, A. R. (1989). Procedures for selecting items for computerized adaptive tests. *Applied Measurement in Education*, 2(4), 359-375.

Magis, D., & Raîche, G. (2012). Random generation of response patterns under computerized adaptive testing with the R package catR. *Journal of Statistical Software*, 48(8), 1-31. <https://doi.org/10.18637/jss.v048.i08>

## Reading the Source

Kingsbury and Zara (1989) was **not checked**: it is not openly available.
Magis and Raîche (2012, p. 12), [open access](https://doi.org/10.18637/jss.v048.i08), describe the randomesque method as one
of the exposure-control methods that select several optimal items near the
current estimate and choose at random among them; catR exposes it as the number
of optimal items to draw from.

**Checked by** `TestRandomesqueSelector` in `tests/test_selection.py`.

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.RandomesqueSelector
   :members:
   :show-inheritance:
```
