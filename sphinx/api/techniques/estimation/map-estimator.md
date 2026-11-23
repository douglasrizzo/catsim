**Status:** Implemented
**Module:** {py:class}`catsim.estimation.MAPEstimator`
**Reference:** Samejima (1969); Bock & Aitkin (1981)

# Maximum A Posteriori Estimator

## Motivation

MAP is the mode-based Bayesian counterpart to MLE. It adds prior information to
the likelihood objective but keeps the optimization workflow familiar, making it
cheaper than quadrature-based EAP while still regularizing extreme short-test
patterns.

This makes MAP the natural option for users who want Bayesian shrinkage without
the overhead of computing posterior moments on every call.

## Definition

The MAP estimate is the posterior mode:

$$
\hat{\theta}_{\mathrm{MAP}}
=
\arg\max_{\theta}\,
p(\theta)\,L(\mathbf{u}\mid\theta)
=
\arg\max_{\theta}\,
\left[\log p(\theta)+\log L(\mathbf{u}\mid\theta)\right].
$$

Equivalently, the estimator minimizes the penalized negative log-posterior

$$
-\log p(\theta) - \log L(\mathbf{u}\mid\theta)
$$

over the configured theta bounds. The estimator first evaluates the
log-posterior at 201 equally spaced points across the bounds (0.06 apart on the
default $[-6, 6]$), in one vectorized call, and takes the best point. A bounded
scalar optimizer then refines the mode between that point's two neighbors, to
within `tol`.

The grid search matters for two reasons. A prior that is zero on part of the
bounds, such as `uniform_log_prior(-1, 1)` searched over $[-6, 6]$, makes the
objective $+\infty$ there, and a bounded scalar search cannot cross that flat
region; it would drift to a bound outside the prior's support. And a multimodal
posterior, from a multimodal prior or a 3PL likelihood, has local modes a local
search can stop at; the grid finds the highest one, provided the posterior does
not change sharply between grid points. If several grid points tie for the
maximum, as with a uniform prior and no responses, the posterior has no unique
mode and the middle of the tied points is returned.

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Parameter
     - Default
     - Description
   * - ``log_prior``
     - ``normal_log_prior(0, 1)``
     - Prior used to regularize the likelihood objective.
   * - ``tol``
     - ``1e-6``
     - Absolute tolerance of the refinement step, passed to the bounded scalar optimizer.
   * - ``bounds``
     - ``(THETA_MIN_EXTENDED, THETA_MAX_EXTENDED)``
     - Search range for the penalized objective.
   * - ``verbose``
     - ``False``
     - Diagnostics flag inherited from the estimator base class.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. With no administered items, the estimate is the mode of the configured prior over the search range, to within `tol`.
2. For short tests with a centered prior, the MAP estimate is closer to the prior mean than the corresponding plain MLE on the same data.
3. Shifting the prior mean while holding the response data fixed shifts the MAP estimate in the same direction.
4. On long, information-rich tests, MAP and EAP should converge toward similar values because the posterior becomes sharply concentrated.
5. The estimator returns finite values on extreme response patterns because the prior keeps the objective bounded.
6. The estimate lies inside the prior's support. A prior with no support inside the search bounds raises `ValueError`.

## References

Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores. *Psychometrika Monograph Supplement*, No. 17.

Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters. *Psychometrika*, 46, 443-459.

:::{seealso}
Related specs: {doc}`/api/techniques/estimation/bayesian-infrastructure`,
{doc}`/api/techniques/estimation/eap-estimator`
:::
