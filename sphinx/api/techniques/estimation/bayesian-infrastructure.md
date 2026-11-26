# Bayesian Estimation Infrastructure

:Class: {py:class}`catsim.estimation.QuadratureGrid`
:Sources: {cite:t}`Bock1982`; {cite:t}`Bock1981`

## Motivation

Bayesian estimators and selectors need the same substrate: a prior on
$\theta$, a fixed quadrature grid, and a numerically stable way to turn
responses into a discrete posterior. This page specifies that shared subsystem
so that later features do not duplicate incompatible posterior code.

The infrastructure is not an estimator by itself. The EAP estimator is built on
it, the MAP estimator reuses its prior factories, and the planned
posterior-weighted selectors (MPWI, KLP, MEPV) would reuse its posterior.

## Definition

For a response vector $\mathbf{u}$ on administered items with parameters
$\boldsymbol{\xi}$, the posterior is

$$
p(\theta \mid \mathbf{u})
=
\frac{p(\theta)\,L(\mathbf{u}\mid\theta)}
     {\int p(\theta)\,L(\mathbf{u}\mid\theta)\,d\theta}.
$$

On a fixed grid $\{\theta_q\}_{q=1}^{Q}$ with quadrature weights
$\{w_q\}_{q=1}^{Q}$, the discrete approximation is

$$
\tilde{p}(\theta_q \mid \mathbf{u})
\propto
w_q\,p(\theta_q)\,L(\mathbf{u}\mid\theta_q),
\qquad
\sum_{q=1}^{Q}\tilde{p}(\theta_q \mid \mathbf{u}) = 1.
$$

The log-likelihoods are summed at each node, as {cite:t}`Bock1982{p. 432}`
recommend for adaptive testing, and the posterior is normalized after
subtracting the largest log-posterior value, so long response vectors do not
underflow. Downstream techniques reuse the same posterior moments:

$$
\hat{\theta}_{\mathrm{EAP}}
=
\sum_{q=1}^{Q} \theta_q \tilde{p}(\theta_q \mid \mathbf{u}),
\qquad
\operatorname{Var}(\theta \mid \mathbf{u})
=
\sum_{q=1}^{Q}(\theta_q-\hat{\theta}_{\mathrm{EAP}})^2
\tilde{p}(\theta_q \mid \mathbf{u}).
$$

These are Equations 4 and 5 of {cite:t}`Bock1982{p. 433}`, without the
square root on the variance; on p. 432 they credit the quadrature formulas to
{cite:t}`Bock1981`. Like Bock and Mislevy, catsim uses evenly spaced nodes
rather than Gauss–Hermite points, which offer little advantage here because the
likelihood is not a polynomial (p. 433).

By default the grid has 121 equally spaced nodes on $[-6, 6]$ (spacing 0.1) and
the prior is $\mathcal{N}(0, 1)$.

### Computation

The grid is accurate while it resolves the posterior. Compared with a
2,401-node reference, the default grid reproduces the posterior mean and
standard deviation to within about 0.25% while the posterior standard deviation
is at least 0.065, which takes a test information of about 240. Below that it
degrades: errors reach about 6% at a posterior standard deviation of 0.05 and
about 44% at 0.035. The default follows the advice of {cite:t}`Bock1982{p. 433}`,
who suggested about 80 points between $\pm 4$ standard deviations, also
a spacing of 0.1, to keep precision down to a posterior standard deviation of
0.2.
A coarser grid degrades sooner: with 41 nodes (spacing 0.3), errors reach about
7% at a posterior standard deviation of 0.15 and about 44% at 0.11.
`EAPEstimator` reports the posterior standard deviation as its standard error,
so a precision stopper with a threshold below about 0.065 needs a finer grid
than the default.

Bock and Mislevy also applied Sheppard's correction to the posterior variance
(p. 433).
catsim does not: the nodes are evaluated rather than grouped, so the
uncorrected moments already match the reference above, and the correction would
shrink the standard deviation by about 1% at a posterior standard deviation of
0.2 and by about 9% at 0.07.

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Parameter
     - Default
     - Description
   * - ``n_nodes``
     - ``121``
     - Number of quadrature nodes of ``QuadratureGrid.uniform``. Consuming
       estimators take a whole grid, so resolution can be tuned without
       changing the shared kernel.
   * - ``low``
     - ``THETA_MIN_EXTENDED``
     - Lower bound of the uniform grid.
   * - ``high``
     - ``THETA_MAX_EXTENDED``
     - Upper bound of the uniform grid.
   * - ``log_prior``
     - ``normal_log_prior(0, 1)``
     - Vectorized log-density used in posterior evaluation.
       ``uniform_log_prior(low, high)`` gives a flat prior instead.
```

## Behavior and Comparison

As informative items accumulate, the posterior variance generally contracts
rather than expands without bound.

A grid's weights may be any positive numbers. catR integrates by the trapezoid
rule, which halves the weights of the two end nodes, and a grid with those
weights gives catR's posterior on the same nodes; the
{doc}`EAP page </api/techniques/estimation/eap-estimator>` shows one.

## Guarantees

1. The posterior probabilities on the quadrature grid sum to 1.
2. The posterior of a long response vector has no NaNs and no all-zero weights from exponentiation underflow.
3. With no responses, the posterior is the prior times the grid weights, normalized.

## References

```{bibliography}
:filter: docname in docnames
```

:::{seealso}
{doc}`/api/techniques/estimation/eap-estimator`,
{doc}`/api/techniques/estimation/map-estimator`
:::

## API Reference

```{eval-rst}
.. autoclass:: catsim.estimation.QuadratureGrid
   :members:
   :show-inheritance:

.. autodata:: catsim.estimation.LogPrior

.. autofunction:: catsim.estimation.normal_log_prior

.. autofunction:: catsim.estimation.uniform_log_prior

.. autofunction:: catsim.estimation.posterior

.. autofunction:: catsim.estimation.posterior_mean

.. autofunction:: catsim.estimation.posterior_variance
```
