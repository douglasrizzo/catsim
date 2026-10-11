**Status:** Implemented
**Module:** {py:class}`catsim.selection.ClusterSelector`
**Reference:** Meneghetti (2015)

# Cluster Selector

## Motivation

Cluster-based selection is an exposure-management strategy that organizes items
into externally supplied groups and chooses from those groups instead of from the
entire bank at once. This makes it possible to distribute usage across groups of
similar items while still preserving an information-based rule inside the chosen
cluster.

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
items. It then chooses an
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
     - Cluster membership label for each item in the bank.
   * - ``method``
     - ``"item_info"``
     - Cluster-scoring rule: maximum item information, sum of cluster
       information, or average cluster information.
   * - ``r_max``
     - ``1.0``
     - Soft maximum exposure rate for choosing an item inside the selected cluster.
   * - ``r_control``
     - ``"passive"``
     - Fallback rule when every remaining item in the chosen cluster violates
       the exposure cap.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. It selects a cluster before selecting an item, and the cluster rule depends only on the configured `method`.
2. The returned item always belongs to the selected cluster and has not been administered previously.
3. If at least one eligible item in the selected cluster has exposure rate below `r_max`, the returned item is the highest-information such item.
4. Under passive control, exposure-cap failure inside the chosen cluster falls back to the highest-information remaining item in that cluster.
5. Under aggressive control, exposure-cap failure inside the chosen cluster falls back to the lowest-exposure remaining item in that cluster.

## References

Meneghetti, D. R. (2015). *Metodologia de selecao de itens em testes adaptativos informatizados baseada em agrupamento por similaridade* (Master's thesis). Centro Universitario da FEI. <https://www.researchgate.net/publication/283944553_Metodologia_de_selecao_de_itens_em_Testes_Adaptativos_Informatizados_baseada_em_Agrupamento_por_Similaridade>

## Reading the Source

Meneghetti (2015) was not checked for this page. The only other description is
one sentence in Meneghetti and Aquino Junior (2017, pp. 9–10,
[arXiv:1707.03012](https://arxiv.org/abs/1707.03012)): the selector "clusters
items according to their parameter values and selects items from the cluster
that contains either the most informative item or the highest average
information". It does not say whether administered items count toward a
cluster's score; this page describes what the code does.

**Checked by** `TestClusterSelector` in `tests/test_selection.py`.

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.ClusterSelector
   :members:
   :show-inheritance:
```
