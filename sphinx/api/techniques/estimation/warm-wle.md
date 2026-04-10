**Status:** Implemented
**Module:** {py:class}`catsim.estimation.WarmLikelihoodEstimator`
**Reference:** Warm (1989)

# Warm Weighted Likelihood Estimator

## Motivation

Maximum-likelihood estimates are biased outward on short tests, which is exactly
the regime encountered in early CAT administration. Warm's weighted likelihood
estimator corrects that first-order bias while staying close to MLE in cost and
asymptotic precision.

This makes WLE a pragmatic upgrade path from plain MLE: it keeps the same basic
optimization workflow, avoids quadrature, and is specifically motivated by the
short-test behavior that matters most in adaptive testing.

## Definition

Warm's estimator maximizes a weighted likelihood proportional to the square root
of Fisher information times the ordinary likelihood. In the closed-form
approximation used here, the estimate is expressed as an MLE plus a bias
correction:

$$
\hat{\theta}_{\mathrm{WLE}}
=
\hat{\theta}_{\mathrm{MLE}}
+
\frac{J(\hat{\theta}_{\mathrm{MLE}})}
     {2\,I(\hat{\theta}_{\mathrm{MLE}})^2},
$$

where $I(\theta)$ is test information and

$$
J(\theta)
=
\sum_i
\frac{P_i'(\theta)\,P_i''(\theta)}
     {P_i(\theta)\,[1-P_i(\theta)]}.
$$

The implementation computes $\hat{\theta}_{\mathrm{MLE}}$ with the
existing numerical optimizer and then applies the analytic correction term.

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Parameter
     - Default
     - Description
   * - ``tol``
     - ``1e-6``
     - Tolerance passed to the underlying numerical MLE routine.
   * - ``verbose``
     - ``False``
     - Diagnostics flag inherited from the estimator base class.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. With no administered items, the estimator returns the input starting value unchanged.
2. For nondegenerate response patterns, the estimate equals the underlying MLE plus the analytic correction $J/(2I^2)$.
3. When the MLE lies away from the population center, the WLE correction moves the estimate back toward the center rather than farther outward.
4. On short tests, WLE should reduce average absolute bias relative to plain MLE under the same response data.
5. If the computed information is nonpositive, the correction term degenerates safely instead of dividing by zero.

## References

Warm, T. A. (1989). Weighted likelihood estimation of ability in item response theory. *Psychometrika*, 54(3), 427-450. <https://doi.org/10.1007/BF02294627>

Lord, F. M. (1983). Unbiased estimators of ability parameters, of their variance, and of their parallel forms reliability. *Psychometrika*, 48(2), 233-245.
