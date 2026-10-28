**Status:** Implemented
**Module:** {py:class}`catsim.selection.The54321Selector`
**Reference:** McBride & Martin (1983)

# 54321 Selector

## Specification

Under maximum-information selection every examinee starts from the same
ability estimate, so the opening items of every test are the same few items.
The 5-4-3-2-1 technique of McBride and Martin (1983) randomizes only those
opening items:

- the first item is drawn uniformly at random from the 5 most informative items
  at the current estimate;
- the second from the 4 most informative remaining items;
- the third from 3 and the fourth from 2;
- from the fifth item on, the most informative remaining item is administered.

This is the schedule described by Georgiadou et al. (2007, p. 9) in their
review of exposure-control strategies. Magis and Raîche (2012, p. 12) list the
technique among the methods that choose at random among several near-optimal
items, without defining it.

With `initial_bin_size` $n$, the item at 0-indexed test position $k$ is drawn
from the $\max(1, n - k)$ most informative non-administered items. Unlike the
randomesque selector, which keeps drawing from a fixed number of items for the
whole test (Georgiadou et al., 2007, p. 9), the late items, which matter most
for precision, are chosen by maximum information.

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

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. It never returns an already administered item.
2. The item at position $k$ is drawn with the configured random number generator from the $\max(1, n - k)$ most informative non-administered items.
3. From position $n - 1$ on, it returns the most informative non-administered item.
4. If no non-administered items remain, it raises `NoItemsAvailableError`.

## References

McBride, J. R., & Martin, J. T. (1983). Reliability and validity of adaptive ability tests in a military setting. In D. J. Weiss (Ed.), *New horizons in testing: Latent trait test theory and computerized adaptive testing* (pp. 223-236). Academic Press. <https://doi.org/10.1016/B978-0-12-742780-5.50022-6>. Not openly available; Georgiadou et al. (2007) describe the method.

Georgiadou, E. G., Triantafillou, E., & Economides, A. A. (2007). A review of item exposure control strategies for computerized adaptive testing developed from 1983 to 2005. *The Journal of Technology, Learning, and Assessment*, 5(8). <https://ejournals.bc.edu/index.php/jtla/article/view/1647> (open access).

Magis, D., & Raîche, G. (2012). Random generation of response patterns under computerized adaptive testing with the R package catR. *Journal of Statistical Software*, 48(8), 1-31. <https://doi.org/10.18637/jss.v048.i08> (open access).

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.The54321Selector
   :members:
   :show-inheritance:
```
