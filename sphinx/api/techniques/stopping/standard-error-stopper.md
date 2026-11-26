# Standard Error Stopper

:Class: {py:class}`catsim.stopping.StandardErrorStopper`
:Sources: {cite:t}`Samejima1977`; {cite:t}`Thissen00`; {cite:t}`Magis12`

## Motivation

Variable-length CATs are usually justified by precision: once the current test
is informative enough, continuing to administer items yields diminishing returns.
The standard-error stopper operationalizes that idea by terminating the test when
standard error of the ability estimate falls to a target value.

This is one of the most common stopping criteria of adaptive tests because its
threshold is easy to interpret on the ability scale and directly reflects the information
accumulated by the current administered set. catR calls it the precision
criterion {cite:p}`Magis12{p. 10}`.

## Definition

The stopper judges the standard error the estimator reported with the current
estimate, the context's `se`
({doc}`ADR-0007 </adr/0007-strategy-contracts>`). It stops when

$$
SE(\hat{\theta}) \le \tau,
$$

where $\tau$ is the configured `max_se`. Each estimator computes that error by
its own formula, as catR does {cite:p}`Magis12{pp. 4-6}`. For
{doc}`maximum likelihood </api/techniques/estimation/numerical-search-estimator>`
and the {doc}`WLE </api/techniques/estimation/warm-wle>` it is the Fisher
standard error, $1/\sqrt{I(\hat{\theta})}$, with $I(\hat{\theta})$ the
information of the administered test at the current estimate. For
{doc}`EAP </api/techniques/estimation/eap-estimator>` it is the posterior
standard deviation, and for
{doc}`MAP </api/techniques/estimation/map-estimator>` the information plus the
prior's negative curvature. Before any response the standard error is
infinite, so the test does not stop. To ask
{meth}`~catsim.stopping.StopRule.satisfied` about a finished session, pass it
{meth}`engine.context(session) <catsim.CatEngine.context>`, which carries the session's
standard error. As in catR, items are
administered "until the standard error of the provisional ability estimate gets
smaller than or equal to that threshold" {cite:p}`Magis12{p. 11}`.

{cite:t}`Samejima1977{p. 238}` suggests terminating an adaptive test when the test
information function reaches a criterion value, or equivalently when the
standard error of estimation falls to a small value, which can be constant
across ability or depend on it; catsim uses a constant. She argues that the
standard error, rather than reliability, should be the index of dependability
(p. 233).

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

## Behavior and Comparison

The EAP and MAP standard errors are usually smaller than the Fisher error at the
same estimate, so the same threshold ends EAP and MAP tests sooner.

A target that some examinees cannot reach with the bank keeps their tests
going until the bank runs out, so the stopper is usually combined with a
{class}`~catsim.stopping.MaxLengthStopper` cap, and often with a minimum length,
in a {class}`~catsim.stopping.StopRule` (see {ref}`combining-stopping-rules`).

### Comparison with catR

The stopper is catR's `precision` rule. The parity tests
({doc}`ADR-0020 </adr/0020-catr-as-the-reference>`) find catR's decisions with
the standard error just above, at and just below the threshold and before any
response, alone and with a length rule, and catR's test lengths in whole tests
on recorded responses with each estimator. They find no departure in the
stopper; the standard error it judges is the estimator's, and the
{doc}`maximum likelihood estimator's page </api/techniques/estimation/numerical-search-estimator>`
states how its error after all-correct or all-incorrect responses departs from
catR's.

## Guarantees

1. The stopper stops the test exactly when the standard error the estimator reported is at most `max_se`.
2. Before any response the standard error is infinite, so the test does not stop.

## References

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. autoclass:: catsim.stopping.StandardErrorStopper
   :members:
   :show-inheritance:
```
