**Status:** Implemented
**Module:** {py:class}`catsim.estimation.EAPEstimator`
**Reference:** Bock & Mislevy (1982)

# Expected A Posteriori Estimator

## Motivation

EAP estimates ability by taking the mean of the posterior distribution rather
than the maximizer of a likelihood. In CAT practice this is attractive because
it always returns a finite value, even for all-correct or all-incorrect patterns
where the MLE does not exist, so no heuristic such as Dodd's is needed early in
the test.

EAP also provides the posterior variance at no extra cost, and that variance
serves as the error variance of the estimate. The same posterior can drive
stopping rules and posterior-weighted selectors.

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

with posterior variance, available through `last_posterior_variance()`,

$$
\operatorname{Var}(\theta \mid \mathbf{u})
\approx
\sum_{q=1}^{Q}
(\theta_q-\hat{\theta}_{\mathrm{EAP}})^2
\tilde{p}(\theta_q \mid \mathbf{u}).
$$

The estimate depends on the prior and on the grid. The default grid (41 nodes
on $[-6, 6]$) is accurate while the posterior standard deviation is at least
about 0.2; longer tests need a finer grid (see
{doc}`/api/techniques/estimation/bayesian-infrastructure`).

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
   * - ``verbose``
     - ``False``
     - Diagnostics flag inherited from the estimator base class.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. With no administered items, the estimate equals the mean of the prior induced by the configured grid and prior.
2. The estimator returns a finite value for extreme response patterns such as all-correct or all-incorrect strings.
3. The exposed posterior from the most recent estimate is normalized and aligned with the configured quadrature grid.
4. Posterior variance should generally decrease as additional informative items are administered.
5. Relative to a flatter prior, a centered normal prior produces stronger shrinkage of extreme short-test estimates toward the prior mean.

## References

Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in a microcomputer environment. *Applied Psychological Measurement*, 6(4), 431-444. <https://doi.org/10.1177/014662168200600405>

Wang, T., & Vispoel, W. P. (1998). Properties of ability estimation methods in computerized adaptive testing. *Journal of Educational Measurement*, 35(2), 109-135.

## Reading the Source

**Checked against** the full text of Bock and Mislevy (1982), openly available
from the [University of Minnesota Digital Conservancy](https://hdl.handle.net/11299/101546).

- **p. 431 (abstract):** EAP is computed noniteratively from sums of
  log-likelihoods at a limited number of quadrature points, and the posterior
  standard deviation behaves like the standard error of measurement.
- **p. 433, Equation 4:** the provisional EAP estimate after each item, as a
  weighted sum over the quadrature points; **Equation 5:** the posterior
  standard deviation.
- **p. 434, Table 1:** an illustrative 20-item session in which the posterior
  standard deviation falls to about 0.25.
- **p. 443:** the conclusion that the EAP estimator is stable at all adaptive
  test lengths, including the first few items.

Wang and Vispoel (1998) compare ML, Bayes modal and EAP estimates in CAT; it
was not checked.

In catsim, `EAPEstimator.estimate` returns Equation 4 and
`last_posterior_variance()` returns the square of Equation 5 (without
Sheppard's correction; see the Bayesian infrastructure page for why).

**Checked by** `TestEAPEstimator` in `tests/test_estimation.py`, including
`test_estimate_returns_prior_mean_with_no_data` and
`test_estimate_converges_toward_mle_on_longer_test`.

:::{seealso}
Depends on {doc}`/api/techniques/estimation/bayesian-infrastructure`
:::

## API Reference

```{eval-rst}
.. autoclass:: catsim.estimation.EAPEstimator
   :members:
   :show-inheritance:
```
