**Status:** Implemented
**Module:** {py:class}`catsim.selection.StratifiedSelector`
**Reference:** Chang & Ying (1999); Chang et al. (2001); Barrada et al. (2006)

# Stratified Selector

## Motivation

Maximum-information selection concentrates exposure on the most discriminating
items and leaves much of the bank unused. Stratified selection controls this by
splitting the bank into strata and the test into matching stages: early stages,
when the ability estimate is still rough, draw from strata of weakly informative
items, and the most informative items are kept for the last stages. Within the
active stratum the selector still adapts, choosing the item located closest to
the current estimate.

{py:class}`~catsim.selection.StratifiedSelector` is the abstract base of the
family. Its subclasses only choose the stratification key, the location matched
to the estimate, and whether to block before stratifying.

## Definition

Let $J$ be the bank size, $L$ the test length (`test_size`) and $S$ the number
of strata (`n_strata`, $1 \le S \le L$). Each subclass defines, for every
item $i$, a stratification key $k_i$ and a location $\ell_i$ on the ability
scale:

| Selector | $k_i$ | $\ell_i$ | Blocking |
| --- | --- | --- | --- |
| {py:class}`~catsim.selection.AStratSelector` | $a_i$ | $b_i$ | no |
| {py:class}`~catsim.selection.AStratBBlockSelector` | $a_i$ | $b_i$ | on $b_i$ |
| {py:class}`~catsim.selection.MaxInfoStratSelector` | $I_i^{\max}$ | $\theta_i^{\max}$ | no |
| {py:class}`~catsim.selection.MaxInfoBBlockSelector` | $I_i^{\max}$ | $\theta_i^{\max}$ | on $\theta_i^{\max}$ |

**Strata without blocking.** Items are sorted by increasing $k_i$ and split
into $S$ contiguous strata of $\lfloor J/S \rfloor$ items, the first
$J \bmod S$ of them holding one item more. This is the a-stratified procedure
of Chang and Ying (1999), as described step by step by Barrada et al. (2006,
p. 157).

**Strata with blocking** (Chang et al., 2001; described by Barrada et al., 2006,
p. 157). Items are sorted by increasing
$\ell_i$ and grouped into consecutive blocks of $S$ items. Each block is
sorted by increasing $k_i$, and its $j$-th item joins stratum $j$. Every full
block therefore gives one item to every stratum, so each stratum spans the
locations those blocks cover, while the strata still ascend in $k_i$. A
trailing block with $r < S$ items, those of highest location, gives its items,
in the same order, to the first $r$ strata, so the strata have the sizes they
would have without blocking.

**Stages.** The test's positions are split, in order, into $S$ stages of
consecutive positions, sized like the strata: $\lfloor L/S \rfloor$ positions
each, the first $L \bmod S$ stages one more. With $q = \lfloor L/S \rfloor$
and $m = L \bmod S$, the item at 0-indexed test position $t < L$ is drawn from
stratum

$$
s(t) =
\begin{cases}
\left\lfloor \dfrac{t}{q + 1} \right\rfloor & \text{if } t < m(q + 1), \\[1ex]
m + \left\lfloor \dfrac{t - m(q + 1)}{q} \right\rfloor & \text{otherwise.}
\end{cases}
$$

Barrada et al. (2006, pp. 157–158) use a bank and a test that divide evenly
into strata; when they do not, catsim gives the extra items and positions to
the first strata and stages. A test of at most $J$ items ($L \le J$) then
never needs more items from a stratum than the stratum holds, so a session
without excluded items administers all $L$ items. There is no stage for
position $L$ or later: asked for an item there, the selector raises
`ValueError`.

**Selection.** Within the active stratum, the selector takes the item whose
location is closest to the estimate (Barrada et al., 2006, p. 157). With $R_t$
the non-administered items,

$$
i^* = \arg\min_{i \in \mathcal{S}_{s(t)} \cap R_t} \left| \ell_i - \hat{\theta} \right|.
$$

The strata depend only on item parameters and are computed once per item bank.

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
     - Intended test length :math:`L`, at least 1. The stopper must end the
       test by then: the selector raises ``ValueError`` when asked for an item
       past it.
   * - ``n_strata``
     - ``None``
     - Number of strata :math:`S`, between 1 and ``test_size``. ``None`` uses
       one stratum per item (:math:`S = L`). The original studies used a few
       strata with several items each: Barrada et al. (2006, pp. 157–158)
       split 250-item banks into 5 strata of 50 items and administered 6
       items from each in a 30-item test.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. The strata partition the bank and ascend in the stratification key; with blocking, every block of `n_strata` items ordered by location contributes its `j`-th item by key to stratum `j`.
2. The active stratum is determined solely by the number of administered items, `test_size` and `n_strata`. Asked for an item once `test_size` items have been administered, the selector raises `ValueError` naming both numbers, so a stopper that allows longer tests than `test_size` fails loudly instead of ending sessions as if the bank were exhausted.
3. The returned item is the non-administered item of the active stratum whose location is closest to the estimate, the context's `theta`.
4. If the active stratum has no non-administered items left, the selector raises `NoItemsAvailableError`.
5. With `n_strata <= test_size <= n_items`, no stage needs more items than its stratum holds, so a session without excluded items administers `test_size` items.
6. The class itself is abstract; subclasses only choose the key, the location and the blocking variable.
7. It applies no exposure cap: it ignores the context's `exposure_rates`. The strata are its exposure control.

## References

Chang, H.-H., & Ying, Z. (1999). A-stratified multistage computerized adaptive testing. *Applied Psychological Measurement*, 23(3), 211-222. <https://doi.org/10.1177/01466219922031338>. Not openly available; Barrada et al. (2006) describe the method.

Chang, H.-H., Qian, J., & Ying, Z. (2001). A-stratified multistage computerized adaptive testing with b blocking. *Applied Psychological Measurement*, 25(4), 333-341. <https://doi.org/10.1177/01466210122032181>. Not openly available; Barrada et al. (2006) describe the method.

Barrada, J. R., Mazuela, P., & Olea, J. (2006). Maximum information stratification method for controlling item exposure in computerized adaptive testing. *Psicothema*, 18(1), 156-159. <https://www.psicothema.com/pdf/3191.pdf> (open access).

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.StratifiedSelector
   :members:
   :show-inheritance:
```
