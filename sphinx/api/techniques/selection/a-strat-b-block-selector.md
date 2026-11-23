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

Following Chang, Qian and Ying (2001), as described by Barrada et al. (2006,
p. 157):

1. Sort all items by increasing $b_i$ and group them into consecutive blocks of
   `n_strata` items.
2. Within each block, sort the items by increasing $a_i$.
3. The $j$-th item of every block joins stratum $j$.

```{figure} /b-blocking.svg
:alt: Blocks 1 to k under an arrow labelled b, each block sorted along its own arrow labelled a. The first items of blocks 1 to k are gathered into the first cell of a new row of strata.
:width: 400px

Items sorted by $b$ into $k$ blocks of `n_strata` items, each block sorted by $a$. The first item of
every block, $\{1_1, 2_1, \dots, k_1\}$, forms the first stratum, the second items the second
stratum, and so on.
```

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
     - Intended test length, at least 1.
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

Chang, H.-H., Qian, J., & Ying, Z. (2001). A-stratified multistage computerized adaptive testing with b blocking. *Applied Psychological Measurement*, 25(4), 333-341. <https://doi.org/10.1177/01466210122032181>. Not openly available; Barrada et al. (2006) describe the method.

Barrada, J. R., Mazuela, P., & Olea, J. (2006). Maximum information stratification method for controlling item exposure in computerized adaptive testing. *Psicothema*, 18(1), 156-159. <https://www.psicothema.com/pdf/3191.pdf> (open access).

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.AStratBBlockSelector
   :members:
   :show-inheritance:
```
