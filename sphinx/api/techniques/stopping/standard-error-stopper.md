**Status:** Implemented
**Module:** {py:class}`catsim.stopping.StandardErrorStopper`
**Reference:** Samejima (1977); Thissen (2000); Magis and Raîche (2012)

# Standard Error Stopper

## Motivation

Variable-length CATs are usually justified by precision: once the current test
is informative enough, continuing to administer items yields diminishing returns.
The standard-error stopper operationalizes that idea by terminating the test when
the standard error of estimation falls to a target value.

This is one of the most common adaptive stopping rules because its threshold is
easy to interpret on the theta scale and directly reflects the information
accumulated by the current administered set. catR calls it the precision
criterion (Magis and Raîche, 2012, p. 10).

## Definition

The stopper judges the standard error the estimator reported with the current
estimate, the context's `se`
({doc}`ADR-0004 </adr/0004-explicit-strategy-contracts>`). It stops when

$$
SE(\hat{\theta}) \le \tau,
$$

where $\tau$ is the configured `max_se`. Each estimator computes that error by
its own formula, as catR does (Magis and Raîche, 2012, pp. 4-6). For
{doc}`maximum likelihood </api/techniques/estimation/numerical-search-estimator>`
and the {doc}`WLE </api/techniques/estimation/warm-wle>` it is the Fisher
standard error, $1/\sqrt{I(\hat{\theta})}$, with $I(\hat{\theta})$ the
information of the administered test at the current estimate. For
{doc}`EAP </api/techniques/estimation/eap-estimator>` it is the posterior
standard deviation, and for
{doc}`MAP </api/techniques/estimation/map-estimator>` the information plus the
prior's negative curvature. Those two are usually smaller than the Fisher error
at the same estimate, so the same threshold ends EAP and MAP tests sooner.
Before any response the standard error is infinite, so the test does not stop. A
context built by hand, such as one for
{meth}`~catsim.stopping.StopRule.satisfied` after a run, must pass the session's
`current_se` as `se`, whose default is infinite. As in catR, items are
administered "until the standard error of the provisional ability estimate gets
smaller than or equal to that threshold" (Magis and Raîche, 2012, p. 11).

Samejima (1977, p. 238) suggests terminating an adaptive test when the test
information function reaches a criterion value, or equivalently when the
standard error of estimation falls to a small value, which can be constant
across ability or depend on it; catsim uses a constant. She argues that the
standard error, rather than reliability, should be the index of dependability
(p. 233).

A target that some examinees cannot reach with the bank keeps their tests
going until the bank runs out, so the stopper is usually combined with a
{class}`~catsim.stopping.MaxLengthStopper` cap, and often with a minimum length,
in a {class}`~catsim.stopping.StopRule` (see {ref}`combining-stopping-rules`).

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``max_se``
     - required
     - Largest standard error :math:`\tau` at which the test stops. Must be positive.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. The stopper returns `True` exactly when the context's standard error, the one the estimator reported, is at most `max_se`.
2. Before any response the standard error is infinite, so the stopper returns `False`.
3. A `max_se` of zero or less raises `ValueError`.

## References

Samejima, F. (1977). A use of the information function in tailored testing. *Applied Psychological Measurement*, 1(2), 233-247. <https://doi.org/10.1177/014662167700100209>. Open copy: [University of Minnesota Digital Conservancy](https://hdl.handle.net/11299/98492).

Thissen, D. (2000). Reliability and measurement precision. In H. Wainer (Ed.), *Computerized Adaptive Testing: A Primer* (2nd ed., pp. 159-184). Mahwah, NJ: Lawrence Erlbaum Associates.

Magis, D., & Raîche, G. (2012). Random generation of response patterns under computerized adaptive testing with the R package catR. *Journal of Statistical Software*, 48(8), 1-31. <https://doi.org/10.18637/jss.v048.i08> (open access).

## API Reference

```{eval-rst}
.. autoclass:: catsim.stopping.StandardErrorStopper
   :members:
   :show-inheritance:
```
