**Status:** Implemented
**Module:** {py:class}`catsim.selection.IntervalInfoSelector`
**Reference:** Veerkamp & Berger (1997)

# Interval Information Selector

## Motivation

Point-information rules assume the current estimate is accurate enough to be
trusted as a single design point, which is least true early in the test.
The interval information criterion of Veerkamp and Berger (1997) instead
integrates each item's information over a confidence interval for $\theta$.
The interval is wide while the estimate is uncertain and narrows as items are
administered, so the rule moves from favoring broadly informative items towards
local maximum-information selection.

## Definition

For an interval $[\theta_L, \theta_U]$, each non-administered item $i \in R_k$
is scored by

$$
S_i = \int_{\theta_L}^{\theta_U} I_i(\theta)\,d\theta,
\qquad
i^* = \arg\max_{i \in R_k} S_i.
$$

By default the interval is the confidence interval of the maximum-likelihood
estimate, using the information $I(\hat{\theta})$ of the administered items:

$$
[\theta_L, \theta_U] = \left[\hat{\theta} - \frac{\lambda}{\sqrt{I(\hat{\theta})}},\;
\hat{\theta} + \frac{\lambda}{\sqrt{I(\hat{\theta})}}\right],
\qquad
\lambda = \Phi^{-1}\left(\frac{1 + \gamma}{2}\right),
$$

where $\gamma$ is the confidence level. Before any item is administered the
interval is the whole ability scale. With a fixed `interval` $\delta$, the
interval is $[\hat{\theta} - \delta, \hat{\theta} + \delta]$ throughout the test.

The integral has a closed form (see {py:func}`catsim.irt.inf_integral`). Since
$dP/d\theta = a (P - c)(d - P)/(d - c)$, substituting $P$ for $\theta$ gives

$$
\int I_i(\theta)\,d\theta =
\frac{a_i}{d_i - c_i}\Big[P - c_i d_i \log P + (1 - c_i)(1 - d_i)\log(1 - P)\Big],
$$

so the scores of all items are computed exactly, without numerical quadrature.
Under the 2PL model, the integral over the whole scale is $a_i$.

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
     - Fixed half-width :math:`\delta` around ``est_theta``. When ``None``, the
       confidence interval above is used.
   * - ``confidence``
     - ``0.95``
     - Confidence level :math:`\gamma` of the default interval.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. Each candidate score is the integral of the item information function over the interval.
2. With `interval=None`, the interval is $\hat{\theta} \pm \lambda / \sqrt{I(\hat{\theta})}$ for the administered items, and the whole ability scale before any item is administered.
3. With a fixed `interval`, the interval is $\hat{\theta} \pm \delta$.
4. It never returns an already administered item.
5. If no non-administered items remain, it raises `NoItemsAvailableError`.

## References

Veerkamp, W. J. J., & Berger, M. P. F. (1997). Some new item selection criteria for adaptive testing. *Journal of Educational and Behavioral Statistics*, 22(2), 203-226. <https://doi.org/10.3102/10769986022002203>

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.IntervalInfoSelector
   :members:
   :show-inheritance:
```
