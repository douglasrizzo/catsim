# Maximum Likelihood Weighted Information Selector

:Class: {py:class}`catsim.selection.MLWISelector`
:Sources: {cite:t}`Veerkamp1997`

## Motivation

MLWI softens local maximum-information selection by weighting each item's
information curve by the likelihood of ability values given the observed
responses. This makes it a frequentist analogue of posterior-weighted selectors:
it spreads attention across plausible ability values without introducing a prior.

## Definition

For candidate item $i$, MLWI is defined as
{cite:p}`Veerkamp1994{pp. 11–12, Equations 17 and 18}`

$$
\mathrm{MLWI}_i
=
\int I_i(\theta)\,L(\mathbf{u}\mid\theta)\,d\theta.
$$

It is the general weighted information criterion of the same report (p. 7,
Equations 7 and 8) with the likelihood as the weight, and it needs no
provisional ability estimate (p. 12). Before any response the likelihood is 1,
so the first item is the one with the most total information (p. 12).
{cite:t}`Magis12{p. 9, Table 1}` give the same criterion.

### Computation

As in Veerkamp and Berger's simulations (p. 15, Equation 20), catsim approximates
the integral on a fixed grid
$\{\theta_q\}_{q=1}^{Q}$ with spacing $\Delta_q$:

$$
\mathrm{MLWI}_i
\approx
\sum_{q=1}^{Q} I_i(\theta_q)\,L(\mathbf{u}\mid\theta_q)\,\Delta_q.
$$

catsim divides the likelihood values by their sum over the grid, which makes them
the posterior that {py:func}`catsim.estimation.posterior` computes under a
uniform prior on `bounds`. That division and the spacing $\Delta_q$, the same
at every node, scale every item's score by one constant, so they do not change
which item is selected.

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Parameter
     - Default
     - Description
   * - ``n_nodes``
     - ``41``
     - Number of integration nodes used in the fixed-grid approximation; at least 5.
   * - ``bounds``
     - ``(THETA_MIN_EXTENDED, THETA_MAX_EXTENDED)``
     - Theta range covered by the fixed integration grid.
   * - ``r_max``
     - ``1.0``
     - Maximum exposure rate, in :math:`(0, 1]`. Items at or above it are left
       out before the weighted information is computed, unless every remaining
       item is.
```

## Behavior and Comparison

In Veerkamp and Berger's simulations MLWI did better than maximum information at
the ML estimate (pp. 17–18). On long tests, where the likelihood is sharply
peaked, its choices approach maximum-information selection near the ML estimate.

:::{dropdown} Comparison with catR

The selector is catR's `nextItem` with `criterion = "MLWI"`, whose default grid,
33 nodes on $[-4, 4]$, is `MLWISelector(n_nodes=33, bounds=(-4, 4))`. catR
integrates by the trapezoid rule, which halves the weights of the two end nodes,
where catsim weights every node equally, as Veerkamp and Berger's sum does; no
`MLWISelector` setting gives trapezoid weights. Before the first response, where
the likelihood is 1 at the end nodes, the scores of the two then differ by up to
2%, a departure the parity tests ({doc}`ADR-0020 </adr/0020-catr-as-the-reference>`)
keep as an expected failure. Summed with equal weights, catR's integrand is in
proportion to catsim's scores, and the tests find catR's ranking of every
remaining item in four contexts and catR's items in whole tests on recorded
responses with the maximum likelihood, Bayes modal, EAP and weighted likelihood
estimators.
:::

## Guarantees

1. Before any response, it selects the item with the most information integrated
   over `bounds`.
2. It never returns an item that was already administered or excluded.
3. When every remaining item is at or above the exposure cap, it falls back to the
   remaining item with the largest weighted information.
4. When no item remains, it raises `NoItemsAvailableError`.

## References

```{bibliography}
:filter: docname in docnames
```

:::{seealso}
{doc}`/api/techniques/selection/mei-selector`, which looks one response ahead
:::

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.MLWISelector
   :members:
   :show-inheritance:
```
