**Status:** Implemented
**Module:** {py:class}`catsim.selection.IntervalInfoSelector`
**Reference:** Veerkamp & Berger (1997)

# Interval Information Selector

## Motivation

Point-information rules assume the current estimate is accurate enough to be
trusted as a single design point, which is least true early in the test.
The interval information criterion of Veerkamp and Berger (1997) instead
integrates each item's information over a confidence interval for $\theta$.
Page and equation numbers below are those of the openly available 1994 research
report that preceded the article; there the criterion is one case of a general
weighted information criterion, $\max_i \int W_n(\theta)\, I_i(\theta)\, d\theta$
(p. 7, Equations 7 and 8).
The interval is wide while the estimate is uncertain and narrows as items are
administered, so the rule moves from favoring broadly informative items towards
local maximum-information selection.

## Definition

For an interval $[\theta_L, \theta_U]$, each non-administered item $i \in R_k$
is scored by

$$
S_i = \int_{\theta_L}^{\theta_U} I_i(\theta)\,d\theta,
\qquad
i^* = \arg\max_{i \in R_k} S_i
$$

(p. 10, Equation 14). Dividing by the interval's length would give a mean
information, but the length is the same for every item.

By default the interval is a confidence interval for $\theta$, whose size
"expresses the uncertainty of the estimator" (p. 9). For the maximum-likelihood
estimate, with $I(\hat{\theta})$ the information of the administered items, it
is

$$
[\theta_L, \theta_U] = \left[\hat{\theta} - \frac{\lambda}{\sqrt{I(\hat{\theta})}},\;
\hat{\theta} + \frac{\lambda}{\sqrt{I(\hat{\theta})}}\right],
\qquad
\lambda = \Phi^{-1}\left(\frac{1 + \gamma}{2}\right),
$$

where $\gamma$ is the confidence level (p. 9, Equation 11). For the EAP
estimate the report uses the posterior standard deviation:

$$
[\theta_L, \theta_U] = \left[\hat{\theta} - \lambda \sqrt{\operatorname{Var}(\theta \mid \mathbf{u})},\;
\hat{\theta} + \lambda \sqrt{\operatorname{Var}(\theta \mid \mathbf{u})}\right]
$$

(p. 9, Equation 12). The report's posterior has a uniform prior (p. 10,
Equation 13); catsim's is the EAP estimator's prior.

catsim builds the interval as $\hat{\theta} \pm \lambda\, SE(\hat{\theta})$, with
the standard error the current estimator reports (the `se` field of
{py:class}`~catsim.context.StepContext`). MLE and WLE sessions therefore get
Equation 11 and EAP sessions Equation 12; a MAP session gets the MAP
estimator's standard error, which includes the prior's curvature, by the same
reasoning. The {doc}`estimation pages </api/techniques/estimation/index>`
define each estimator's standard error.

Before the first response the standard error is infinite and the interval is
the whole ability scale (p. 11, Equation 16). The report keeps this unbounded
interval while the total score is zero or perfect; catsim keeps it only while
the standard error is infinite. With a fixed `interval` $\delta$, the interval
is $[\hat{\theta} - \delta, \hat{\theta} + \delta]$ throughout the test.

The integral has a closed form (see {py:func}`catsim.irt.info_integral`). Since
$dP/d\theta = a (P - c)(d - P)/(d - c)$, substituting $P$ for $\theta$ gives

$$
\int I_i(\theta)\,d\theta =
\frac{a_i}{d_i - c_i}\Big[P - c_i d_i \log P + (1 - c_i)(1 - d_i)\log(1 - P)\Big],
$$

so the scores of all items are computed exactly, without numerical quadrature.
This extends the 3PL closed form of Veerkamp and Berger (pp. 21–22, Appendix,
Equation 25), to which it reduces when $d_i = 1$. Under the 2PL model, the
integral over the whole scale is $a_i$ (Equation 28).

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``interval``
     - ``None``
     - Fixed half-width :math:`\delta` around the estimate. When ``None``, the
       confidence interval above is used.
   * - ``confidence``
     - ``0.95``
     - Confidence level :math:`\gamma` of the default interval.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. Each candidate score is the integral of the item information function over the interval.
2. With `interval=None`, the interval is $\hat{\theta} \pm \lambda\, SE(\hat{\theta})$, with the standard error the current estimator reports, and the whole ability scale while that standard error is infinite, as it is before the first response.
3. With a fixed `interval`, the interval is $\hat{\theta} \pm \delta$.
4. It never returns an already administered item.
5. If no non-administered items remain, it raises `NoItemsAvailableError`.
6. It applies no exposure cap: it ignores the context's `exposure_rates`.

## References

Veerkamp, W. J. J., & Berger, M. P. F. (1997). Some new item selection criteria for adaptive testing. *Journal of Educational and Behavioral Statistics*, 22(2), 203-226. <https://doi.org/10.3102/10769986022002203>. Not openly available; the 1994 report below is the open version.

Veerkamp, W. J. J., & Berger, M. P. F. (1994). *Some new item selection criteria for adaptive testing* (Research Report 94-6). Enschede: University of Twente. Open copy: [University of Twente](https://ris.utwente.nl/ws/files/5155670/RR_94-6.pdf).

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.IntervalInfoSelector
   :members:
   :show-inheritance:
```
