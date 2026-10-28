**Status:** Implemented
**Module:** {py:class}`catsim.selection.RandomesqueSelector`

# Randomesque Selector

## Specification

`RandomesqueSelector` ranks non-administered items by Fisher information at
the current ability estimate, keeps the top `bin_size` candidates, and then
selects uniformly at random from that candidate set.

The technique relaxes maximum-information selection by adding controlled
randomness among the most informative items. Unlike the
{doc}`5-4-3-2-1 technique <the-54321-selector>`, it keeps drawing from the same
number of best items for the whole test instead of switching to maximum
information (Georgiadou et al., 2007, p. 9). Magis and Raîche (2012, p. 12) list
it among the exposure-control methods that choose at random among several
optimal items near the current estimate.

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. It considers only non-administered items.
2. It builds its random candidate set from the most informative remaining items.
3. It samples from the candidate set using the configured random number generator.
4. If no non-administered items remain, it raises `NoItemsAvailableError`.

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
