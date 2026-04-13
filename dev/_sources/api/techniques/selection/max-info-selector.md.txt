**Status:** Implemented
**Module:** {py:class}`catsim.selection.MaxInfoSelector`
**Reference:** Lord (1980); Samejima (1977)

# Maximum Fisher Information Selector

## Motivation

Maximum Fisher information is the canonical local item-selection rule in CAT. At
each step it chooses the remaining item that is most informative at the current
ability estimate $\hat{\theta}$, which makes it the natural baseline for both
efficiency and implementation simplicity.

Its main weakness is that it trusts $\hat{\theta}$ completely. Early in a test,
when the estimate is still unstable, the rule can overuse highly discriminating
items and produce skewed exposure patterns. The selector remains valuable because
many other criteria in the literature are best understood as relaxations or
extensions of this rule.

## Definition

For each non-administered item $i$, the selector evaluates Fisher
information at the current estimate:

$$
I_i(\hat{\theta})
=
\frac{a_i^2 [P_i(\hat{\theta})-c_i]^2 [d_i-P_i(\hat{\theta})]^2}
     {(d_i-c_i)^2 P_i(\hat{\theta}) [1-P_i(\hat{\theta})]}.
$$

The selected item is

$$
i^*
=
\arg\max_{i \in R_k} I_i(\hat{\theta}),
$$

where $R_k$ is the set of non-administered items at step $k$. In the
package implementation, items are first ordered by descending information; any
item whose exposure rate is below `r_max` is preferred, but if every remaining
item exceeds the cap the selector falls back to the highest-information item
anyway.

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
       to the unrestricted maximum-information choice.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. It orders candidate items by decreasing Fisher information at the current `est_theta`.
2. It never returns an already administered item.
3. If at least one non-administered item has exposure rate below `r_max`, the returned item comes from that subset.
4. If all remaining items violate the exposure cap, it still returns the highest-information remaining item.
5. If no non-administered items remain, it raises `NoItemsAvailableError`.

## References

Lord, F. M. (1980). *Applications of Item Response Theory to Practical Testing Problems*. Hillsdale, NJ: Lawrence Erlbaum Associates.

Samejima, F. (1977). A use of the information function in tailored testing. *Applied Psychological Measurement*, 1(2), 233-247. <https://doi.org/10.1177/014662167700100209>

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.MaxInfoSelector
   :members:
   :show-inheritance:
```
