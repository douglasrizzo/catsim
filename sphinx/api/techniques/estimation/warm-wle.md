# Warm Weighted Likelihood Estimator

:Class: {py:class}`catsim.estimation.WarmLikelihoodEstimator`
:Sources: {cite:t}`Warm1989`

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
$\frac{d}{d\theta}\ln w(\theta) = J(\theta) / [2 I(\theta)]$
{cite:p}`Warm1985{p. 8, Equations 9 and 10}`, so the estimate is the root of the modified score equation

$$
\sum_i \frac{\bigl[u_i - P_i(\theta)\bigr] P_i'(\theta)}{P_i(\theta)\,[1-P_i(\theta)]}
+
\frac{J(\theta)}{2\,I(\theta)}
= 0,
$$

{cite:p}`Warm1985{p. 8, Equation 11}`; {cite:p}`Magis12{p. 6, Equations 12 and 13}`,
where $I(\theta)$ is test information and

$$
J(\theta)
=
\sum_i
\frac{P_i'(\theta)\,P_i''(\theta)}
     {P_i(\theta)\,[1-P_i(\theta)]}.
$$

For models without guessing or slipping ($c = 0$, $d = 1$), $J = I'$ and the
weight is $w(\theta) = \sqrt{I(\theta)}$: the WLE then maximizes
$L(\theta)\sqrt{I(\theta)}$, the likelihood under a Jeffreys prior
{cite:p}`Warm1985{p. 9}`; {cite:p}`Magis12{p. 6}`. With guessing
or slipping, $J \neq I'$, so the WLE is defined by the equation above and not by
the $\sqrt{I}$-weighted likelihood.

Expanding the equation around the MLE gives the familiar approximation
$\hat{\theta}_{\mathrm{WLE}} \approx \hat{\theta}_{\mathrm{MLE}} + J/(2I^2)$, which
is why $-J/(2I^2)$ is the MLE's first-order bias {cite:p}`Lord1983`, as given by
{cite:t}`Warm1985{p. 7, Equation 6}`. The implementation
does not use this approximation: it fails exactly where WLE matters most, because
the MLE is infinite for all-correct and all-incorrect patterns, and with guessing it
can sit on the search bound, where $I \approx 0$ makes the correction explode.

The estimator returns a {py:class}`~catsim.Estimate`. Its standard error
is the Fisher one at the estimate, $1/\sqrt{I(\hat{\theta})}$, infinite where
the information is 0. This departs from {cite:t}`Magis12{p. 6, Equation 14}`,
who add $(I'J - IJ')/(2I^2)$ to the information. Under the 1PL and 2PL models the
corrected sum is the curvature of the weighted likelihood at its maximum, so it is
positive; with guessing or slipping parameters it can be negative, which leaves no
standard error: 13 of 3,920 sets of responses (0.4% of their probability) on random
3PL tests of 2 to 8 items, none of 4,560 on 2PL tests.

### Computation

The estimate is irtlib's, `irtlib.ability.weighted_likelihood`. The modified score is evaluated at
121 equally spaced points over the search range $[\theta_{\min}, \theta_{\max}]$ given by
`bounds`, by default $[-6, 6]$
(`THETA_MIN_EXTENDED`, `THETA_MAX_EXTENDED`), to bracket its roots. On the
default range the points are 0.1 apart, so two roots closer than that can fall
in one interval and go unseen. Each sign change from positive to negative brackets a maximum
of the weighted likelihood and is refined with Brent's method. If several maxima
exist, which can happen with guessing, the one with the largest weighted
likelihood is returned; the maxima are ranked by integrating the modified score,
the derivative of the weighted log-likelihood. If the modified score never changes
sign, the estimate is the bound it points to.

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
     - Absolute tolerance of the Brent root refinement. Must be positive.
   * - ``bounds``
     - ``(THETA_MIN_EXTENDED, THETA_MAX_EXTENDED)``
     - Search range of the estimate. Must be finite, with the lower bound below the upper one.
```

## Behavior and Comparison

Against the WLE's sampling spread at a known ability (every possible set of
responses to 5 and 10 items, 3,000 simulated ones for 20 and 40), the classic formula
overestimates the error in short tests, by 10% under the 2PL and 28% under the 3PL
at 5 items. That is conservative: a standard-error stopper runs a few items longer
than needed. Equation 14 roughly halves that gap. The sandwich formula of
{cite:t}`Magis2016` is closest under the 3PL but underestimates the
error under the 2PL (by 8% at 5 items), which would stop tests too early. From 20
items on, the three agree within about 3%.

:::{dropdown} Comparison with catR

The estimator is catR's `thetaEst` with `method = "WL"`, and its standard error
is that of catR's `semTheta` with its default `sem.type = "classic"`, which
catR has given the WLE since version 3.0, because the corrected formula sometimes
led to mathematical problems ([catR's
NEWS](https://github.com/cran/catR/blob/master/NEWS)); its `sem.type = "new"` is
the sandwich formula of {cite:t}`Magis2016`. With
`bounds=(-4, 4)`, catR's default range, the parity tests
({doc}`ADR-0020 </adr/0020-catr-as-the-reference>`) find the same estimates and
standard errors on fixed responses, all-correct and all-incorrect ones
included, and in whole tests on recorded responses under MFI, bOpt and MLWI
selection, with one departure, which they keep as an expected failure.

When the modified score has several roots in the range, catR's `uniroot` returns
whichever root its search converges to, and catsim the one of largest weighted
likelihood (see Computation); no setting makes catsim follow catR. Two items
with guessing, one answered correctly and the other incorrectly, can give three
roots, two maxima and a minimum: on one such pattern catR returns −0.59 and
catsim 0.66. In a whole test with MLWI selection, the roots after the first two
items are −0.44, 0.63 and 0.84, and catR returns 0.84 and catsim −0.44; the test's
items are the same in both, and the estimates agree again from the third
response on.
:::

## Guarantees

1. Every interior estimate is a root of the modified score equation, including for items with guessing.
2. For items without guessing or slipping, the estimate maximizes the likelihood under a Jeffreys prior.
3. The estimate is finite and inside the search interval for any responses, including
   all-correct and all-incorrect patterns where the MLE diverges.
4. Relative to a finite MLE, the WLE is pulled toward the abilities at which the administered items are most informative.
5. On a short test without guessing, the exact expected bias (by enumeration of every possible set of responses)
   is small, and smaller than the MLE's first-order bias above wherever that bias is
   non-negligible. With guessing and few items, the WLE at low ability is pulled upward, toward
   the items, so no unconditional bias guarantee is made for that case.
6. The standard error is the Fisher standard error at the returned estimate, or `math.inf` where the information is 0.

## References

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. autoclass:: catsim.estimation.WarmLikelihoodEstimator
   :members:
   :show-inheritance:
```
