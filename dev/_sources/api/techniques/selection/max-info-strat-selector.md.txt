**Status:** Implemented
**Module:** {py:class}`catsim.selection.MaxInfoStratSelector`
**Reference:** Barrada et al. (2006)

# Maximum-Information Stratified Selector

## Motivation

Maximum-information stratification (MIS) adapts the stratified-exposure idea to
item pools where the 3PL or 4PL information maximum is not captured well by
difficulty alone. Instead of reserving high-discrimination items for later
stages, it reserves items with larger *maximum attainable information* for later
stages.

This makes MIS a closer relative of the maximum-information selector than the
original a-stratified family, while still moderating exposure through stagewise
access to the bank.

## Definition

For each item $i$, let

$$
I_i^{\max} = \max_{\theta} I_i(\theta).
$$

The package implementation first presorts items by increasing
$I_i^{\max}$. After partitioning that order into stage strata, it postsorts
items *within the active stratum* by current information
$I_i(\hat{\theta})$ in descending order, and returns the first
non-administered item.

Equivalently, at position $k+1$, selection is

$$
i^*
=
\arg\max_{i \in S_k \cap R_k} I_i(\hat{\theta}),
$$

where $S_k$ is the active MIS stratum induced by ordering items on
$I_i^{\max}$.

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``test_size``
     - required
     - Number of strata and intended test length.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. Strata are formed from a presort by increasing maximum attainable item information.
2. The active stratum is resorted on every selection step by information at the current `est_theta`.
3. The returned item maximizes current information within the active stratum among non-administered items.
4. It never returns an already administered item.
5. If the active stratum has been exhausted, it raises `NoItemsAvailableError`.

## References

Barrada, J. R., Mazuela, P., & Olea, J. (2006). Maximum information stratification method for controlling item exposure in computerized adaptive testing. *Psicothema*, 18(1), 156-159. <https://pubmed.ncbi.nlm.nih.gov/17296025/>

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.MaxInfoStratSelector
   :members:
   :show-inheritance:
```
