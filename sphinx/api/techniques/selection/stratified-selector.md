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
into $S$ contiguous strata whose sizes differ by at most one item.

**Strata with blocking** (Chang et al., 2001). Items are sorted by increasing
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

**Selection.** With $R_t$ the non-administered items,

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
       strata with several items each, for example 5 strata for a 30-item test
       (Barrada et al., 2006).
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. The strata partition the bank and ascend in the stratification key; with blocking, every block of `n_strata` items ordered by location contributes its `j`-th item by key to stratum `j`.
2. The active stratum is determined solely by the number of administered items, `test_size` and `n_strata`, and stays at the last stratum past `test_size`.
3. The returned item is the non-administered item of the active stratum whose location is closest to `est_theta`.
4. If the active stratum has no non-administered items left, the selector raises `NoItemsAvailableError`.
5. The class itself is abstract; subclasses only choose the key, the location and the blocking variable.

## References

Chang, H.-H., & Ying, Z. (1999). A-stratified multistage computerized adaptive testing. *Applied Psychological Measurement*, 23(3), 211-222. <https://doi.org/10.1177/01466219922031338>

Chang, H.-H., Qian, J., & Ying, Z. (2001). A-stratified multistage computerized adaptive testing with b blocking. *Applied Psychological Measurement*, 25(4), 333-341. <https://doi.org/10.1177/01466210122032181>

Barrada, J. R., Mazuela, P., & Olea, J. (2006). Maximum information stratification method for controlling item exposure in computerized adaptive testing. *Psicothema*, 18(1), 156-159. <https://pubmed.ncbi.nlm.nih.gov/17296025/>

## Reading the Source

Chang and Ying (1999) and Chang, Qian and Ying (2001) were **not checked**: they
are not openly available. All four methods were checked against their
step-by-step description in the full text of Barrada, Mazuela and Olea (2006),
open access from [Psicothema](https://www.psicothema.com/pdf/3191.pdf):

- **p. 157, a-stratification** (attributed to Chang & Ying, 1999): choose the
  number of strata, the items per stratum and the items administered from each
  (summing to the bank size and the test length); sort the bank by increasing
  $a$ and fill the strata in that order; administer the first items from
  stratum 1, the next from stratum 2, and so on, each time selecting the item
  that minimizes $|\hat{\theta} - b|$.
- **p. 157, b-blocking** (attributed to Chang, Qian & Ying, 2001): divide the
  bank, sorted by $b$, into blocks; sort each block by increasing $a$; the first
  item of every block forms stratum 1, the second items stratum 2, and so on.
  Selection then follows the a-stratified rule. The purpose is to give every
  stratum a balanced distribution of $b$ when $a$ and $b$ are correlated.
- **p. 157, Equations 1 and 2, maximum information stratification:** replace
  $a$ by the item's maximum information when stratifying, and $b$ by the
  ability where that maximum is reached, both in the selection rule and as the
  blocking variable. The authors' formulas include the scaling constant
  $D = 1.7$; catsim's IRT functions use no scaling constant.
- **pp. 157–158, simulation design:** 250-item banks, 30-item tests, 5 strata
  of 50 items with 6 items administered from each, a starting estimate drawn
  from $U(-0.5, 0.5)$, and Dodd's (1990) heuristic until the response pattern
  is mixed.
- **pp. 158–159:** blocking reduced the skew of the exposure rates whether or
  not $a$ and $b$ were correlated, and the authors recommend MIS with blocking
  whenever a stratified method is used.

In catsim, `strata` builds the strata and `select` applies the stage rule. The
number of strata defaults to one per test item, a special case of the
published designs, which use a few strata with several items each.

**Checked by** `TestStratifiedSelectors` in `tests/test_selection.py`.

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.StratifiedSelector
   :members:
   :show-inheritance:
```
