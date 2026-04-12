**Status:** Planned
**Module:** `catsim.estimation.bayesian.EAPEstimator` (planned)
**Reference:** Bock & Mislevy (1982)

# Expected A Posteriori Estimator

## Motivation

EAP estimates ability by taking the mean of the posterior distribution rather
than the maximizer of a likelihood. In CAT practice this is attractive because
it always returns a finite value, even for all-correct or all-incorrect patterns
where MLE becomes unstable or undefined.

EAP also exposes a posterior variance naturally, which makes it useful beyond
estimation itself: the same posterior can drive stopping rules and
posterior-weighted selectors. That makes it a foundational Bayesian estimator
for the planned feature set.

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

Using the shared quadrature grid from the Bayesian infrastructure, the planned
implementation approximates this as

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

The estimator therefore depends on the prior and the quadrature grid, but it
avoids the divergence issues that motivate Dodd-style MLE fallbacks.

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
     - Planned diagnostics flag inherited from the estimator base class.
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

:::{seealso}
Depends on {doc}`/api/techniques/estimation/bayesian-infrastructure`
:::
