# Maximum Fisher Information Selector

:Class: {py:class}`catsim.selection.MaxInfoSelector`
:Sources: {cite:t}`Lord77`; {cite:t}`Lord1980`

## Motivation

Maximum Fisher information is the canonical local item-selection rule in CAT. At
each step it chooses the remaining item that is most informative at the current
ability estimate $\hat{\theta}$, which makes it the natural baseline for both
efficiency and implementation simplicity. {cite:t}`Lord77{p. 97}` describes the
rule: after each response the ability is re-estimated, and the next item is "the
item that gives the most information at his estimated ability level".

## Definition

For each item $i$ that is neither administered nor excluded, the selector
evaluates the item information $I_i(\hat{\theta})$ at the current estimate:
the general item information of {cite:t}`Magis12{p. 4, Equation 4}`, which
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

:::{dropdown} Comparison with catR

The selector is catR's `nextItem` with `criterion = "MFI"`, and it breaks ties as
catR does ({ref}`selection-ties`). The parity tests
({doc}`ADR-0020 </adr/0020-catr-as-the-reference>`) find catR's information values
and its ranking of every remaining item in four contexts, from before the first
response to an all-correct pattern, and catR's items in whole tests on recorded
responses with the maximum likelihood, Bayes modal, EAP and weighted likelihood
estimators. They find no departure. The exposure cap differs from catR's `rmax`
({ref}`selection-exposure-cap`).
:::

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

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.MaxInfoSelector
   :members:
   :show-inheritance:
```
