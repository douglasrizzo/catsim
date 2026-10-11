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

## Reading the Source

**Checked against** the 1994 research report that preceded the journal article,
openly available from the
[University of Twente](https://ris.utwente.nl/ws/files/5155670/RR_94-6.pdf)
(Veerkamp & Berger, *Some new item selection criteria for adaptive testing*,
Research Report 94-6). The 1997 journal version was not checked. Page numbers
are the report's own.

- **p. 7, Equations 7 and 8:** the general weighted information criterion,
  $\max_i \int W_n(\theta)\, I_i(\theta)\, d\theta$.
- **p. 9, Equation 11:** the confidence interval for the ML estimate,
  $\hat{\theta} \pm \lambda\, I(\hat{\theta})^{-1/2}$ with
  $\lambda = \Phi^{-1}[(1 + \gamma)/2]$; **Equation 12** gives the EAP version,
  with the posterior standard deviation in place of $I^{-1/2}$.
- **p. 10, Equation 14:** the interval information criterion, the area under
  the item information function over that interval. Dividing by the interval's
  length would give a mean, but the length is the same for every item.
- **p. 11, Equations 15 and 16:** the criterion as a weight of 1 inside the
  interval and 0 outside, and the starting interval $(-\infty, \infty)$ before
  any item has been administered.
- **pp. 21–22, Appendix, Equations 24–28:** 3PL information and the closed-form
  integral
  $\int_{\theta_L}^{\theta_R} I_i\, d\theta = \frac{a_i}{1 - c_i}\big[c_i \ln\frac{P_i(\theta_L)}{P_i(\theta_R)} + P_i(\theta_R) - P_i(\theta_L)\big]$
  (Equation 25), which reduces to $a_i[P_i(\theta_R) - P_i(\theta_L)]$ for
  the 2PL model (Equation 26) and to $a_i$ over the whole scale (Equation 28).

catsim's `irt.inf_integral` extends Equation 25 to the 4PL model; with $d = 1$
it is algebraically identical to Equation 25. The default interval always uses
Equation 11, with the test information at the current estimate, whatever
estimator produced it.

**Checked by** `TestIntervalInfoSelector` in `tests/test_selection.py` and
`TestInfIntegral` in `tests/test_irt.py`, which compare the closed form with
numerical integration.

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.IntervalInfoSelector
   :members:
   :show-inheritance:
```
