# Expected A Posteriori Estimator

:Class: {py:class}`catsim.estimation.EAPEstimator`
:Sources: {cite:t}`Bock1982`; {cite:t}`Bock1981`

## Motivation

EAP estimates ability by taking the mean of the posterior distribution rather
than the maximizer of a likelihood. In CAT practice this is attractive because
it always returns a finite value, even for all-correct or all-incorrect patterns
where the MLE does not exist, so no heuristic such as Dodd's is needed early in
the test; {cite:t}`Bock1982{p. 443}` conclude that it is stable at all
adaptive test lengths, including the first few items.

EAP's posterior also yields the posterior variance at no extra cost, and that
variance serves as the error variance of the estimate {cite:p}`Bock1982{p. 432}`;
in their simulations the posterior standard deviation behaved like the
standard error of measurement (p. 431). The estimator reports the posterior
standard deviation as the estimate's standard error
{cite:p}`Magis12{p. 6, Equation 11}`, and the precision stoppers judge it.

## Definition

The EAP estimate is the posterior mean:

$$
\hat{\theta}_{\mathrm{EAP}}
=
\mathbb{E}[\theta \mid \mathbf{u}]
=
\frac{\int \theta\,p(\theta)\,L(\mathbf{u}\mid\theta)\,d\theta}
     {\int p(\theta)\,L(\mathbf{u}\mid\theta)\,d\theta}.
$$

On a grid of nodes $\theta_q$, $q = 1, \dots, Q$, with weights $w_q$, irtlib's `QuadratureGrid`,
the posterior mass at a node is proportional to $w_q\,p(\theta_q)\,L(\mathbf{u}\mid\theta_q)$ and
the masses $\tilde{p}(\theta_q \mid \mathbf{u})$ sum to 1. The implementation approximates the
integral as

$$
\hat{\theta}_{\mathrm{EAP}}
\approx
\sum_{q=1}^{Q}\theta_q\,\tilde{p}(\theta_q \mid \mathbf{u}),
$$

with posterior variance

$$
\operatorname{Var}(\theta \mid \mathbf{u})
\approx
\sum_{q=1}^{Q}
(\theta_q-\hat{\theta}_{\mathrm{EAP}})^2
\tilde{p}(\theta_q \mid \mathbf{u}).
$$

These are Equations 4 and 5 of {cite:t}`Bock1982{p. 433}`, who credit the quadrature formulas to
{cite:t}`Bock1981` (p. 432); catsim returns the variance, the square of their Equation 5, without
the Sheppard's correction they apply (p. 433). Like Bock and Mislevy, catsim uses evenly spaced
nodes rather than Gauss–Hermite points, which offer little advantage here because the likelihood is
not a polynomial (p. 433).

The estimator returns a {py:class}`~catsim.Estimate` with
$\hat{\theta}_{\mathrm{EAP}}$ and the posterior standard deviation
$\sqrt{\operatorname{Var}(\theta \mid \mathbf{u})}$ as its standard error. With
no administered item the posterior is the prior, so the estimate and its
standard error are the prior's mean and standard deviation on the grid.

### Computation

By default the grid has 121 equally spaced nodes on $[-6, 6]$ (spacing 0.1) and the prior is
$\mathcal{N}(0, 1)$. The grid is accurate while it resolves the posterior. Compared with a
2,401-node reference, the default grid reproduces the posterior mean and standard deviation to
within about 0.25% while the posterior standard deviation is at least 0.065, which takes a test
information of about 240. Below that it degrades: errors reach about 6% at a posterior standard
deviation of 0.05 and about 44% at 0.035. The default follows the advice of
{cite:t}`Bock1982{p. 433}`, who suggested about 80 points between $\pm 4$ standard deviations, also
a spacing of 0.1, to keep precision down to a posterior standard deviation of 0.2. A coarser grid
degrades sooner: with 41 nodes (spacing 0.3), errors reach about 7% at a posterior standard
deviation of 0.15 and about 44% at 0.11. The estimator reports the posterior standard deviation as
its standard error, so a precision stopper with a threshold below about 0.065 needs a finer grid
than the default.

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Parameter
     - Default
     - Description
   * - ``grid``
     - ``QuadratureGrid.uniform()``
     - Discrete integration grid used to approximate posterior moments.
   * - ``log_prior``
     - ``normal_log_prior(0, 1)``
     - Prior density on :math:`\theta`; changing it changes both shrinkage and zero-data behavior.
```

## Behavior and Comparison

The estimate depends on the prior and on the grid. The default grid (121 nodes
on $[-6, 6]$) is accurate while the posterior standard deviation is at least
about 0.065; longer tests need a finer grid (see Computation above).

:::{dropdown} Comparison with catR

The estimator is catR's `thetaEst` with `method = "EAP"`, and its standard error
is that of catR's `semTheta`, the posterior standard deviation. catR integrates
by the trapezoid rule on its `parInt` grid, by default 33 nodes on $[-4, 4]$,
which halves the weights of the two end nodes; catsim's default grid has 121
equally weighted nodes on $[-6, 6]$
({doc}`ADR-0020 </adr/0020-catr-as-the-reference>`). A grid with catR's nodes and
weights reproduces catR's estimator:

```python
import numpy
from catsim.estimation import EAPEstimator
from irtlib.quadrature import QuadratureGrid

weights = numpy.full(33, 8 / 32)
weights[[0, -1]] /= 2
catr_eap = EAPEstimator(grid=QuadratureGrid(nodes=numpy.linspace(-4, 4, 33), weights=weights))
```

On such grids the parity tests find catR's estimates and standard errors, to a
relative $10^{-10}$, on fixed responses under normal and uniform priors,
on catsim's 121 nodes as on catR's 33, and in whole tests on recorded responses.
The simulations that check the randomized selectors against catR estimate with
this estimator too. They find no departure.
:::

## Guarantees

1. The estimate is finite for all-correct and all-incorrect responses.
2. The standard error is the posterior standard deviation on the grid.
3. Relative to a flatter prior, a centered normal prior produces stronger shrinkage of extreme short-test estimates toward the prior mean.

## References

```{bibliography}
:filter: docname in docnames

Wang1998
```

## API Reference

```{eval-rst}
.. autoclass:: catsim.estimation.EAPEstimator
   :members:
   :show-inheritance:
```
