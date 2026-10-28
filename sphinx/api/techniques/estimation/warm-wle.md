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

Warm's estimator maximizes the likelihood multiplied by a weight function
$w(\theta)$ chosen so that the first-order ($O(n^{-1})$) bias of the MLE cancels.
The weight is defined through its log-derivative,
$\frac{d}{d\theta}\ln w(\theta) = J(\theta) / [2 I(\theta)]$ (Warm, 1985, p. 8,
Equations 9 and 10), so the estimate is the root of the modified score equation

$$
\sum_i \frac{\bigl[u_i - P_i(\theta)\bigr] P_i'(\theta)}{P_i(\theta)\,[1-P_i(\theta)]}
+
\frac{J(\theta)}{2\,I(\theta)}
= 0,
$$

(Warm, 1985, p. 8, Equation 11; Magis & Raîche, 2012, p. 6, Equations 12 and
13), where $I(\theta)$ is test information and

$$
J(\theta)
=
\sum_i
\frac{P_i'(\theta)\,P_i''(\theta)}
     {P_i(\theta)\,[1-P_i(\theta)]}.
$$

For models without guessing or slipping ($c = 0$, $d = 1$), $J = I'$ and the
weight is $w(\theta) = \sqrt{I(\theta)}$: the WLE then maximizes
$L(\theta)\sqrt{I(\theta)}$, the likelihood under a Jeffreys prior (Warm, 1985,
p. 9; Magis & Raîche, 2012, p. 6). With guessing
or slipping, $J \neq I'$, so the WLE is defined by the equation above and not by
the $\sqrt{I}$-weighted likelihood.

Expanding the equation around the MLE gives the familiar approximation
$\hat{\theta}_{\mathrm{WLE}} \approx \hat{\theta}_{\mathrm{MLE}} + J/(2I^2)$, which
is why $-J/(2I^2)$ is the MLE's first-order bias (Lord, 1983, as given by Warm,
1985, p. 7, Equation 6). The implementation
does not use this approximation: it fails exactly where WLE matters most, because
the MLE is infinite for all-correct and all-incorrect patterns, and with guessing it
can sit on the search bound, where $I \approx 0$ makes the correction explode.

### Computation

The modified score is evaluated on a grid over
$[\theta_{\min}, \theta_{\max}]$ (`THETA_MIN_EXTENDED`, `THETA_MAX_EXTENDED`) to
bracket its roots. Each sign change from positive to negative brackets a maximum
of the weighted likelihood and is refined with Brent's method. If several maxima
exist, which can happen with guessing, the one with the largest weighted
likelihood is returned; candidates are ranked by integrating the modified score,
the derivative of the weighted log-likelihood. If the modified score never changes
sign, the estimate is the bound it points to. This mirrors `catR` (Magis and
Raîche, 2012), whose
[`thetaEst.R`](https://github.com/cran/catR/blob/master/R/thetaEst.R) solves the
same equation with `uniroot`; catsim adds the explicit handling of multiple
roots. Magis
and Raîche (2012, p. 6, Equation 14) also give the WLE's standard error, which
catsim does not compute.

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
     - Absolute tolerance of the Brent root refinement.
   * - ``verbose``
     - ``False``
     - Diagnostics flag inherited from the estimator base class.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. With no administered items, the estimator returns the input starting value unchanged.
2. Every interior estimate is a root of the modified score equation, including for items with guessing.
3. For items without guessing or slipping, the estimate maximizes $L(\theta)\sqrt{I(\theta)}$.
4. The estimate is finite and inside the search interval for every response pattern, including
   all-correct and all-incorrect patterns where the MLE diverges.
5. Relative to a finite MLE, the WLE is pulled toward the abilities at which the administered items are most informative.
6. On a short test without guessing, the exact expected bias (by enumeration of all response patterns)
   is small, and smaller than the MLE's first-order bias $-J/(2I^2)$ wherever that bias is
   non-negligible. With guessing and few items, the WLE at low ability is pulled upward, toward
   the items, so no unconditional bias guarantee is made for that case.

## References

Warm, T. A. (1989). Weighted likelihood estimation of ability in item response theory. *Psychometrika*, 54(3), 427-450. <https://doi.org/10.1007/BF02294627>. Not openly available; Warm (1985) derives the same estimator.

Warm, T. A. (1985). *Weighted likelihood estimation of ability in item response theory with tests of finite length* (Doctoral dissertation). University of Oklahoma. <https://shareok.org/handle/11244/5356> (open access).

Lord, F. M. (1983). Unbiased estimators of ability parameters, of their variance, and of their parallel-forms reliability. *Psychometrika*, 48(2), 233-245. <https://doi.org/10.1007/BF02294018>. Not openly available; Warm (1985, p. 7) gives its bias formula.

Magis, D., & Raîche, G. (2012). Random generation of response patterns under computerized adaptive testing with the R package catR. *Journal of Statistical Software*, 48(8), 1-31. <https://doi.org/10.18637/jss.v048.i08> (open access).

## API Reference

```{eval-rst}
.. autoclass:: catsim.estimation.WarmLikelihoodEstimator
   :members:
   :show-inheritance:
```
