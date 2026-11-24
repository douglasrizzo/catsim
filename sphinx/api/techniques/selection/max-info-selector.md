# Maximum Fisher Information Selector

:Class: {py:class}`catsim.selection.MaxInfoSelector`
:Sources: Lord (1977); Lord (1980)

## Motivation

Maximum Fisher information is the canonical local item-selection rule in CAT. At
each step it chooses the remaining item that is most informative at the current
ability estimate $\hat{\theta}$, which makes it the natural baseline for both
efficiency and implementation simplicity. Lord (1977, p. 97) describes the
rule: after each response the ability is re-estimated, and the next item is "the
item that gives the most information at his estimated ability level".

## Definition

For each item $i$ that is neither administered nor excluded, the selector
evaluates the item information $I_i(\hat{\theta})$ at the current estimate:
the general item information of Magis and Raîche (2012, p. 4, Equation 4), which
{py:func}`catsim.irt.information` computes under the 4PL. The selected item is

$$
i^*
=
\arg\max_{i \in R_k} I_i(\hat{\theta}),
$$

where $R_k$ is the set of remaining items at step $k$. With an exposure cap
`r_max` below 1, any item whose exposure rate is below `r_max` is preferred, but
if every remaining item is at or above the cap the selector falls back to the
highest-information item anyway.

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``r_max``
     - ``1.0``
     - Soft maximum exposure rate. Remaining items with exposure rate
       :math:`< r_{\max}` are preferred; if none exist, the selector falls back
       to the unrestricted maximum-information choice. The default of 1 applies
       no cap.
```

## Behavior and Comparison

Its main weakness is that it trusts $\hat{\theta}$ completely. Early in a test,
when the estimate is still unstable, the rule can overuse highly discriminating
items and produce skewed exposure patterns. The selector remains valuable because
many other criteria in the literature are best understood as relaxations or
extensions of this rule.

## Guarantees

1. Without an exposure cap, it returns the most informative remaining item at the
   current estimate.
2. It never returns an item that was already administered or excluded.
3. If `r_max` is below 1 and at least one remaining item has an exposure rate
   below `r_max`, the returned item comes from that subset.
4. If every remaining item is at or above the exposure cap, it still returns the
   highest-information remaining item.
5. When no item remains, it raises `NoItemsAvailableError`.

## References

Lord, F. M. (1977). A broad-range tailored test of verbal ability. *Applied Psychological Measurement*, 1(1), 95-100. <https://doi.org/10.1177/014662167700100115>. Open copy: [University of Minnesota Digital Conservancy](https://hdl.handle.net/11299/98423).

Lord, F. M. (1980). *Applications of Item Response Theory to Practical Testing Problems*. Hillsdale, NJ: Lawrence Erlbaum Associates.

Magis, D., & Raîche, G. (2012). Random generation of response patterns under computerized adaptive testing with the R package catR. *Journal of Statistical Software*, 48(8), 1-31. <https://doi.org/10.18637/jss.v048.i08> (open access).

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.MaxInfoSelector
   :members:
   :show-inheritance:
```
