**Status:** Implemented
**Module:** {py:class}`catsim.selection.ClusterSelector`
**Reference:** Meneghetti (2015)

# Cluster Selector

## Motivation

The method comes from Meneghetti (2015). Cluster-based selection is an
exposure-management strategy that organizes items into externally supplied
groups and chooses from those groups instead of from the entire bank at once.
The aim is to distribute usage across groups of similar items while still
preserving an information-based rule inside the chosen cluster. In the
simulations reported below under "How It Compares", it did not improve on
simpler selectors.

In this package the selector is explicitly operational rather than purely
theoretical: users supply the cluster memberships, choose how cluster quality is
scored, and choose how to behave when all items in the chosen cluster exceed the
exposure cap.

## Definition

Let $C_g$ denote cluster $g$, and let $R_k$ denote the
remaining items. The package supports three cluster-scoring rules:

$$
S_g^{\mathrm{item}}(\hat{\theta})
=
\max_{i \in C_g} I_i(\hat{\theta}),
$$

$$
S_g^{\mathrm{cluster}}(\hat{\theta})
=
\sum_{i \in C_g} I_i(\hat{\theta}),
$$

$$
S_g^{\mathrm{weighted}}(\hat{\theta})
=
\frac{1}{|C_g|}\sum_{i \in C_g} I_i(\hat{\theta}).
$$

All three scores use every item of the cluster, including items already
administered. The selector first picks the best cluster according to the
configured scoring rule among clusters that still contain non-administered
items; clusters whose scores tie exactly are chosen between uniformly at random,
as tied items are (see {doc}`/adr/0008-random-tie-breaking`). It then chooses an
item inside that cluster:

$$
i^*
=
\arg\max_{i \in C_{g^*} \cap R_k,\ r_i < r_{\max}} I_i(\hat{\theta})
$$

whenever such items exist. If none do, `r_control="passive"` falls back to the
highest-information item in the cluster, whereas `r_control="aggressive"`
chooses the lowest-exposure item in the cluster.

Two consequences follow. Because a cluster keeps its score after its best items
are administered, the selector can return to a cluster whose remaining items are
much less informative than items elsewhere in the bank. And because the cluster
is chosen without looking at exposure rates, `r_max` only applies inside the
chosen cluster; it does not cap exposure across the bank.

Meneghetti and Aquino Junior (2018, pp. 9–10) describe the selector as one that
"clusters items according to their parameter values and selects items from the
cluster that contains either the most informative item or the highest average
information". They do not say whether administered items count toward a
cluster's score; this page describes what the code does.

## How It Compares

Simulations run while preparing catsim 1.0 measured those consequences. They
used two 500-item 3PL banks, clusters from k-means on the standardized item
parameters, 1,000 examinees drawn from N(0, 1) and 20-item tests, and compared
the selector with `MaxInfoSelector` and `RandomesqueSelector`. Test information
was measured at the true ability. The first bank was clustered with k = 10, 25
and 50, the second with k = 25.

- Without an exposure cap, `item_info` gave less test information and more test
  overlap than `MaxInfoSelector`: 5.21 to 5.59 against 5.82, with overlap 0.408
  to 0.422 against 0.369, on the first bank, and 4.98 against 5.77, with overlap
  0.404 against 0.359, on the second.
- `cluster_info` lost the most information on the first bank (4.30 to 5.42),
  because a sum grows with the size of the cluster: 23% to 47% of its picks had
  less than half the information of the best remaining item.
- `weighted_info` lowered overlap a little, but `RandomesqueSelector(bin_size=5)`
  kept more information at lower overlap on both banks: on the first,
  `weighted_info` reached 5.42 to 5.64 at an overlap of 0.331 to 0.366, and
  `RandomesqueSelector` 5.66 at 0.320.
- With `r_max = 0.2` and k = 25, and exposure rates computed over the examinees
  tested so far, 35 to 65 items ended above the cap for `item_info` and
  `weighted_info` under both `r_control` values, with peak rates from 0.28 to
  0.90. `MaxInfoSelector(r_max=0.2)` left no item above it.

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``clusters``
     - required
     - One non-negative integer cluster label for each item in the bank. The
       number of labels is checked against the bank at every selection.
   * - ``method``
     - ``ClusterMethod.ITEM_INFO``
     - Cluster-scoring rule, a ``ClusterMethod`` or its string value:
       ``"item_info"`` (maximum item information), ``"cluster_info"`` (sum of
       cluster information) or ``"weighted_info"`` (average cluster information).
   * - ``r_max``
     - ``1.0``
     - Soft maximum exposure rate for choosing an item inside the selected cluster.
   * - ``r_control``
     - ``CapFallback.PASSIVE``
     - Fallback rule when every remaining item in the chosen cluster violates
       the exposure cap, a ``CapFallback`` or its string value
       (``"passive"`` or ``"aggressive"``).
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. It selects a cluster before selecting an item, and the cluster rule depends only on the configured `method`.
2. The returned item always belongs to the selected cluster and has not been administered previously.
3. If at least one eligible item in the selected cluster has exposure rate below `r_max`, the returned item is the highest-information such item.
4. Under passive control, exposure-cap failure inside the chosen cluster falls back to the highest-information remaining item in that cluster.
5. Under aggressive control, exposure-cap failure inside the chosen cluster falls back to the lowest-exposure remaining item in that cluster.
6. Clusters whose scores tie exactly, and items tied inside the chosen cluster, are drawn uniformly at random.
7. It raises `ValueError` if `clusters` does not hold one label per item of the bank.

## References

Meneghetti, D. R. (2015). *Metodologia de selecao de itens em testes adaptativos informatizados baseada em agrupamento por similaridade* (Master's thesis). Centro Universitario da FEI. <https://www.researchgate.net/publication/283944553_Metodologia_de_selecao_de_itens_em_Testes_Adaptativos_Informatizados_baseada_em_Agrupamento_por_Similaridade>

Meneghetti, D. D. R., & Aquino Junior, P. T. (2018). *Application and simulation of computerized adaptive tests through the package catsim* (arXiv:1707.03012v2). <https://arxiv.org/abs/1707.03012> (open access).

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.ClusterSelector
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: catsim.selection.ClusterMethod
   :members:
```

```{eval-rst}
.. autoclass:: catsim.selection.CapFallback
   :members:
```
