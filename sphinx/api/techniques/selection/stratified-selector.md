# Stratified Selectors

:Class: {py:class}`catsim.selection.AStratSelector`
:Class: {py:class}`catsim.selection.AStratBBlockSelector`
:Class: {py:class}`catsim.selection.MaxInfoStratSelector`
:Class: {py:class}`catsim.selection.MaxInfoBBlockSelector`
:Class: {py:class}`catsim.selection.StratifiedSelector`
:Sources: {cite:t}`Chang1999`; {cite:t}`Chang2001`; {cite:t}`Barrada2006`

## Motivation

Maximum-information selection concentrates exposure on the most discriminating
items and leaves much of the bank unused. Stratified selection controls this by
splitting the bank into strata and the test into matching stages: early stages,
when the ability estimate is still rough, draw from strata of weakly informative
items, and the most informative items are kept for the last stages. Within the
active stratum the selector still adapts, choosing the item located closest to
the current estimate.

The a-stratified method of {cite:t}`Chang1999` stratifies by discrimination and
matches difficulty to the estimate. Item discrimination and difficulty are often
positively correlated, so strata formed on $a$ alone inherit skewed difficulty
distributions: early strata lack hard items and late strata lack easy ones,
which hurts both precision and exposure balance. The b-blocking refinement of
{cite:ts}`Chang2001` forces every stratum to cover the whole difficulty
range before discrimination-based stratification is applied.

The a-stratified method ignores the guessing parameter, both when stratifying
and when selecting. Under the 3PL and 4PL models an item's discrimination does
not rank its maximum information, and its information does not peak at $b$.
Maximum-information stratification (MIS) of {cite:t}`Barrada2006` stratifies by
the item's maximum information and matches the ability at which that maximum is
reached. MIS-B applies the blocking idea to MIS: it blocks on that ability, so
that every stratum covers the whole ability range.

The four selectors share the base class
{py:class}`~catsim.selection.StratifiedSelector` and differ only in the
stratification key, the location matched to the estimate, and whether they block
before stratifying.

## Definition

Let $M$ be the bank size, $n$ the test length (`test_size`) and $S$ the number
of strata (`n_strata`, $1 \le S \le n$). Each selector defines, for every
item $i$, a stratification key $k_i$ and a location $\ell_i$ on the ability
scale:

| Selector | Method | $k_i$ | $\ell_i$ | Blocking |
| --- | --- | --- | --- | --- |
| {py:class}`~catsim.selection.AStratSelector` | a-stratified {cite:p}`Chang1999` | $a_i$ | $b_i$ | no |
| {py:class}`~catsim.selection.AStratBBlockSelector` | a-stratified with b-blocking {cite:p}`Chang2001` | $a_i$ | $b_i$ | on $b_i$ |
| {py:class}`~catsim.selection.MaxInfoStratSelector` | MIS {cite:p}`Barrada2006` | $I_i^{\max}$ | $\theta_i^{\max}$ | no |
| {py:class}`~catsim.selection.MaxInfoBBlockSelector` | MIS-B {cite:p}`Barrada2006` | $I_i^{\max}$ | $\theta_i^{\max}$ | on $\theta_i^{\max}$ |

MIS and MIS-B use the ability at which item $i$ is most informative and its
information there:

$$
\theta_i^{\max} = \arg\max_{\theta} I_i(\theta),
\qquad
I_i^{\max} = I_i(\theta_i^{\max}).
$$

{py:func}`catsim.irt.max_info_theta` computes $\theta_i^{\max}$.

{cite:t}`Barrada2006{p. 157, Equations 1 and 2}` give $I_i^{\max}$ and
$\theta_i^{\max}$ in closed form for the 3PL model, with the scaling constant
$D = 1.7$; catsim's IRT functions use no scaling constant.

**Strata without blocking.** Items are sorted by increasing $k_i$ and split
into $S$ contiguous strata of $\lfloor M/S \rfloor$ items, the first
$M \bmod S$ of them holding one item more. This is the a-stratified procedure
of {cite:t}`Chang1999`, as described step by step by
{cite:t}`Barrada2006{p. 157}`.

```{figure} /alpha-strat.svg
:alt: A row of strata numbered 1 to k under an arrow labelled a, shading darker from stratum 1 to stratum k.
:width: 400px

`AStratSelector`: the item bank sorted by increasing $a$ and split into strata
$1, \dots, k$.
```

```{figure} /mis.svg
:alt: A row of strata numbered 1 to k under an arrow labelled maximum information, shading darker from stratum 1 to stratum k.
:width: 400px

`MaxInfoStratSelector`: the item bank sorted by increasing $I_i^{\max}$ and split
into strata $1, \dots, k$.
```

**Strata with blocking** {cite:p}`Chang2001`;
{cite:p}`{described by}Barrada2006{p. 157}`. Items are sorted by increasing
$\ell_i$ and grouped into consecutive blocks of $S$ items. Each block is sorted
by increasing $k_i$; its first item joins the first stratum, its second item
the second stratum, and so on. Every
full block therefore gives one item to every
stratum, so each stratum spans the locations those blocks cover, while the
strata still ascend in $k_i$. A trailing block of $r < S$ items, those of
highest location, gives its items, in the same order, to the first $r$ strata, so
the strata have the sizes they would have without blocking.

```{figure} /b-blocking.svg
:alt: Blocks 1 to k under an arrow labelled b, each block sorted along its own arrow labelled a. The first items of blocks 1 to k are gathered into the first cell of a new row of strata.
:width: 400px

`AStratBBlockSelector`: items sorted by $b$ into $k$ blocks of $S$ items, each
block sorted by $a$. The first item of every block, $\{1_1, 2_1, \dots, k_1\}$,
forms the first stratum, the second items the second stratum, and so on.
```

```{figure} /mis-b.svg
:alt: Blocks 1 to k under an arrow labelled theta max, each block sorted along its own arrow labelled maximum information. The first items of blocks 1 to k are gathered into the first cell of a new row of strata.
:width: 400px

`MaxInfoBBlockSelector`: items sorted by $\theta_i^{\max}$ into $k$ blocks of $S$
items, each block sorted by $I_i^{\max}$. The first item of every block,
$\{1_1, 2_1, \dots, k_1\}$, forms the first stratum, the second items the second
stratum, and so on.
```

**Stages.** The test's positions are split, in order, into $S$ stages of
consecutive positions, sized like the strata: $\lfloor n/S \rfloor$ positions
each, the first $n \bmod S$ stages one more. With $q = \lfloor n/S \rfloor$
and $h = n \bmod S$, the item at 0-indexed test position $t < n$ is drawn from
stratum

$$
s(t) =
\begin{cases}
\left\lfloor \dfrac{t}{q + 1} \right\rfloor & \text{if } t < h(q + 1), \\[1ex]
h + \left\lfloor \dfrac{t - h(q + 1)}{q} \right\rfloor & \text{otherwise.}
\end{cases}
$$

{cite:t}`Barrada2006{pp. 157–158}` use a bank and a test that divide evenly
into strata; when they do not, catsim gives the extra items and positions to
the first strata and stages. A test of at most $M$ items ($n \le M$) then
never needs more items from a stratum than the stratum holds, so a session
without excluded items administers all $n$ items. There is no stage for
position $n$ or later: asked for an item there, the selector raises
`ValueError`.

**Selection.** Within the active stratum, the selector takes the item whose
location is closest to the estimate {cite:p}`Barrada2006{p. 157}`. With $R_t$
the items neither administered nor excluded,

$$
i^* = \arg\min_{i \in \mathcal{S}_{s(t)} \cap R_t} \left| \ell_i - \hat{\theta} \right|.
$$

The strata depend only on the item parameters, so every session of a run uses
the same strata.

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
     - Intended test length :math:`n`, at least 1. The stopper must end the
       test by then: the selector raises ``ValueError`` when asked for an item
       past it.
   * - ``n_strata``
     - ``None``
     - Number of strata :math:`S`, between 1 and ``test_size``. ``None`` uses
       one stratum per test item (:math:`S = n`). The original studies used a
       few strata with several items each: :cite:t:`Barrada2006{pp. 157–158}`
       split 250-item banks into 5 strata of 50 items and administered 6
       items from each in a 30-item test.
```

## Behavior and Comparison

Under the 1PL and 2PL models $\theta_i^{\max} = b_i$ and $I_i^{\max}$ ranks
items like $a_i$, so MIS coincides with the a-stratified method.

MIS improved precision over the a-stratified method in
{cite:t}`Barrada2006{pp. 158–159}`. They also found that blocking reduced the
skewness of the exposure rates both with and without correlation between $a$
and $b$, and recommended MIS-B whenever a stratified method is used.

## Guarantees

1. The strata, which `strata(item_bank)` returns, partition the bank and ascend
   in the stratification key; with blocking, every block of `n_strata` items
   ordered by location gives its items, in order of key, one to each stratum, so
   every stratum spans the bank's range of locations.
2. Each item comes from the stratum of its stage, which depends only on the
   number of administered items, `test_size` and `n_strata`: the first stages
   draw from the strata of lowest key, the last stage from the highest.
3. The returned item is the remaining item of the active stratum whose location
   is closest to the estimate; it is never an item already administered or
   excluded.
4. If the active stratum has no remaining items, the selector raises
   `NoItemsAvailableError`.
5. With `n_strata <= test_size <= n_items`, no stage needs more items than its
   stratum holds, so a session without excluded items administers `test_size`
   items.
6. Asked for an item once `test_size` items have been administered, the selector
   raises `ValueError` naming both numbers, so a stopper that allows longer tests
   than `test_size` fails loudly rather than end sessions as if the bank were
   exhausted.
7. It applies no exposure cap: the run's exposure rates do not change its choice.
   The strata are its exposure control.

## References

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.StratifiedSelector
   :members:
   :show-inheritance:

.. autoclass:: catsim.selection.AStratSelector
   :members:
   :show-inheritance:

.. autoclass:: catsim.selection.AStratBBlockSelector
   :members:
   :show-inheritance:

.. autoclass:: catsim.selection.MaxInfoStratSelector
   :members:
   :show-inheritance:

.. autoclass:: catsim.selection.MaxInfoBBlockSelector
   :members:
   :show-inheritance:
```
