**Status:** Implemented
**Module:** {py:class}`catsim.selection.AStratBBlockSelector`
**Reference:** Chang et al. (2001)

# A-Stratified with B-Blocking Selector

## Motivation

Item discrimination and difficulty are often positively correlated. Strata
formed on $a$ alone then inherit skewed difficulty distributions: early strata
lack hard items and late strata lack easy ones, which hurts both precision and
exposure balance. The b-blocking refinement forces every stratum to cover the
whole difficulty range before discrimination-based stratification is applied.

## Definition

Following Chang, Qian and Ying (2001):

1. Sort all items by increasing $b_i$ and group them into consecutive blocks of
   `n_strata` items.
2. Within each block, sort the items by increasing $a_i$.
3. The $j$-th item of every block joins stratum $j$.

Each stratum therefore contains one item from every difficulty block, and the
strata ascend in discrimination. Selection then follows the a-stratified rule:
at test position $t$, return the non-administered item of stratum $s(t)$ whose
difficulty is closest to the current estimate,

$$
i^* = \arg\min_{i \in \mathcal{S}_{s(t)} \cap R_t} \left| b_i - \hat{\theta} \right|.
$$

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
     - Intended test length.
   * - ``n_strata``
     - ``None``
     - Number of strata and stages; ``None`` uses ``test_size``.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. In every block of `n_strata` items ordered by difficulty, the `j`-th item by discrimination belongs to stratum `j`.
2. Every stratum spans the difficulty range of the bank.
3. The returned item minimizes $|b_i - \hat{\theta}|$ among the non-administered items of the active stratum.
4. It never returns an already administered item.
5. If the active stratum has no non-administered items left, it raises `NoItemsAvailableError`.

## References

Chang, H.-H., Qian, J., & Ying, Z. (2001). A-stratified multistage computerized adaptive testing with b blocking. *Applied Psychological Measurement*, 25(4), 333-341. <https://doi.org/10.1177/01466210122032181>

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.AStratBBlockSelector
   :members:
   :show-inheritance:
```
