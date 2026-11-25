# Expected A Posteriori Estimator

:Class: {py:class}`catsim.estimation.EAPEstimator`
:Sources: {cite:t}`Bock1982`

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
standard deviation as the estimate's standard error, as catR does
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

Using the shared quadrature grid from the
{doc}`Bayesian infrastructure </api/techniques/estimation/bayesian-infrastructure>`,
the implementation approximates this as

$$
\hat{\theta}_{\mathrm{EAP}}
\approx
\sum_{q=1}^{Q}\theta_q\,\tilde{p}(\theta_q \mid \mathbf{u}),
$$

with posterior variance, computed by `posterior_variance()`,

$$
\operatorname{Var}(\theta \mid \mathbf{u})
\approx
\sum_{q=1}^{Q}
(\theta_q-\hat{\theta}_{\mathrm{EAP}})^2
\tilde{p}(\theta_q \mid \mathbf{u}).
$$

These are Equations 4 and 5 of {cite:t}`Bock1982{p. 433}`; catsim returns
the variance, the square of their Equation 5, without Sheppard's correction (see
{doc}`/api/techniques/estimation/bayesian-infrastructure`).

The estimator returns a {py:class}`~catsim.Estimate` with
$\hat{\theta}_{\mathrm{EAP}}$ and the posterior standard deviation
$\sqrt{\operatorname{Var}(\theta \mid \mathbf{u})}$ as its standard error. With
no administered item the posterior is the prior, so the estimate and its
standard error are the prior's mean and standard deviation on the grid.

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
about 0.065; longer tests need a finer grid (see
{doc}`/api/techniques/estimation/bayesian-infrastructure`).

## Guarantees

1. The estimate is finite for extreme response patterns such as all-correct or all-incorrect strings.
2. The standard error is the posterior standard deviation on the grid.
3. Relative to a flatter prior, a centered normal prior produces stronger shrinkage of extreme short-test estimates toward the prior mean.

## References

```{bibliography}
:filter: docname in docnames

Wang1998
```

:::{seealso}
Depends on {doc}`/api/techniques/estimation/bayesian-infrastructure`
:::

## API Reference

```{eval-rst}
.. autoclass:: catsim.estimation.EAPEstimator
   :members:
   :show-inheritance:
```
