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
into $S$ contiguous strata whose sizes differ by at most one item. This is the
a-stratified procedure of Chang and Ying (1999), as described step by step by
Barrada et al. (2006, p. 157).

**Strata with blocking** (Chang et al., 2001; described by Barrada et al., 2006,
p. 157). Items are sorted by increasing
$\ell_i$ and grouped into consecutive blocks of $S$ items. Each block is
sorted by increasing $k_i$, and its $j$-th item joins stratum $j$. Every
stratum therefore spans the whole range of locations, while the strata still
ascend in $k_i$. A trailing block with $r < S$ items sends its $j$-th item
to stratum $\lfloor jS/r \rfloor$.

**Stages.** The item at 0-indexed test position $t$ is drawn from stratum

$$
s(t) = \min\left(\left\lfloor \frac{tS}{L} \right\rfloor,\; S - 1\right),
$$

so each stratum serves about $L/S$ consecutive positions, and a session that
runs past $L$ items stays in the last stratum.

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
     - Intended test length :math:`L`.
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
2. The active stratum is determined solely by the number of administered items, `test_size` and `n_strata`, and stays at the last stratum past `test_size`.
3. The returned item is the non-administered item of the active stratum whose location is closest to `est_theta`.
4. If the active stratum has no non-administered items left, the selector raises `NoItemsAvailableError`.
5. The class itself is abstract; subclasses only choose the key, the location and the blocking variable.

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
