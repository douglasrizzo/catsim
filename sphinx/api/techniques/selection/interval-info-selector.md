# Interval Information Selector

:Class: {py:class}`catsim.selection.IntervalInfoSelector`
:Sources: {cite:t}`Veerkamp1997`

## Motivation

Point-information rules assume the current estimate is accurate enough to be
trusted as a single design point, which is least true early in the test.
The interval information criterion of {cite:t}`Veerkamp1997` integrates
each item's information over a confidence interval for $\theta$ rather than
evaluate it at a single point.
Page and equation numbers below are those of the openly available {cite:year}`Veerkamp1994`
research report that preceded the article; there the criterion is one case of a general
weighted information criterion, $\max_i \int W_n(\theta)\, I_i(\theta)\, d\theta$
(p. 7, Equations 7 and 8).
The interval is wide while the estimate is uncertain and narrows as items are
administered, so the rule moves from favoring broadly informative items towards
local maximum-information selection.

## Definition

For an interval $[\theta_L, \theta_U]$, each remaining item $i \in R_k$ (neither
administered nor excluded)
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

### Computation

catsim builds the interval as $\hat{\theta} \pm \lambda\, SE(\hat{\theta})$, with
the standard error the session's estimator reports. MLE and WLE sessions therefore get
Equation 11 and EAP sessions Equation 12; a MAP session gets the MAP
estimator's standard error, which includes the prior's curvature, by the same
reasoning. The {doc}`estimation pages </api/techniques/estimation/index>`
define each estimator's standard error.

Before the first response the standard error is infinite and the interval is
the whole ability scale (p. 11, Equation 16). The report keeps this unbounded
interval while the total score is zero or perfect; catsim keeps it only while
the standard error is infinite. A standard error so small that the interval's
half-width is below about $10^{-8}$ collapses the interval onto $\hat{\theta}$;
catsim then selects by the items' information at $\hat{\theta}$, the limit of the
information averaged over the interval. With a
fixed `interval` $\delta$, the interval is $[\hat{\theta} - \delta, \hat{\theta} + \delta]$
throughout the test.

The integral has a closed form, which {py:func}`catsim.irt.info_integral`
computes, so the scores of all items are computed exactly, without numerical
quadrature. It extends the 3PL closed form of
{cite:t}`Veerkamp1994{pp. 21–22, Appendix, Equation 25}`, to which it reduces
when $d_i = 1$. Under the 2PL model,
the integral over the whole scale is $a_i$ (Equation 28).

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

## Behavior and Comparison

With the default interval, the selector judges the estimate by the standard error
its estimator reports, so the same responses usually give a narrower interval, and
more local choices, under EAP or MAP than under maximum likelihood (see the
{doc}`estimation index </api/techniques/estimation/index>`).

## Guarantees

1. It returns the remaining item whose information, integrated over the interval,
   is largest.
2. It never returns an item that was already administered or excluded.
3. When no item remains, it raises `NoItemsAvailableError`.
4. It applies no exposure cap: the exposure rates do not change its choice.

## References

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.IntervalInfoSelector
   :members:
   :show-inheritance:
```
