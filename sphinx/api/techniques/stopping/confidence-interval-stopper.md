# Confidence Interval Stopper

:Class: {py:class}`catsim.stopping.ConfidenceIntervalStopper`
:Sources: {cite:t}`Thissen00`; {cite:t}`Wainer2000`

## Motivation

Some adaptive tests are used for categorical decisions rather than for reporting a
single continuous ability estimate. In that setting it is often more natural to
stop once the uncertainty around $\hat{\theta}$ falls entirely within one
predefined ability band, because the classification is then stable.

The confidence-interval stopper in {mod}`catsim` implements exactly that
classification-oriented precision rule. An examinee whose ability lies close to a
boundary may never be classified, so the stopper is usually combined with a
{class}`~catsim.stopping.MaxLengthStopper` cap in a {class}`~catsim.stopping.StopRule`
(see {ref}`combining-stopping-rules`).

## Definition

The package computes a normal-approximation confidence interval

$$
CI(\hat{\theta})
=
\left[
  \hat{\theta} - z_{(1+\gamma)/2} \, SE(\hat{\theta}),
  \hat{\theta} + z_{(1+\gamma)/2} \, SE(\hat{\theta})
\right],
$$

where $\gamma$ is the configured confidence level, $z_{(1+\gamma)/2}$ the
standard normal quantile, and $SE(\hat{\theta})$ the standard error the
estimator reported with the current estimate, the context's `se`
({doc}`ADR-0007 </adr/0007-strategy-contracts>`). Each estimator
computes it by its own formula: the Fisher standard error
$1/\sqrt{I(\hat{\theta})}$ for maximum likelihood and the WLE, the posterior
standard deviation for EAP, and for MAP the information plus the prior's
negative curvature (see {doc}`/api/techniques/stopping/standard-error-stopper`).

Given sorted boundary points $b_1 < \dots < b_m$, the rule stops when the
entire confidence interval falls within one interval induced by those boundaries:

$$
(-\infty, b_1],\ [b_1, b_2],\ \dots,\ [b_m, \infty).
$$

Equivalently, the interval lies within one band exactly when no boundary lies
strictly inside it, which is what the implementation checks. Before any response
the standard error is infinite, and so is the interval, so the test does not
stop. To ask {meth}`~catsim.stopping.StopRule.satisfied` about a finished
session, pass it {meth}`engine.context(session) <catsim.CatEngine.context>`, which
carries the session's standard error.

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``interval_bounds``
     - required
     - Sorted boundary points that define the ability bands used for classification.
   * - ``confidence``
     - ``0.95``
     - Confidence level :math:`\gamma` used to form the normal-approximation interval.
```

## Behavior and Comparison

A smaller standard error narrows the interval, so EAP and MAP sessions are
usually classified sooner than they would be on the Fisher error.

{cite:t}`Magis12{p. 11}` describe the classification criterion with a
single ability threshold: a confidence interval for ability is built after every
item, and the test stops once the threshold is no longer inside it. catsim
generalizes this to several boundaries; with one boundary the two rules agree.

### Comparison with catR

With one boundary the stopper is catR's `classification` rule, with
`confidence` equal to 1 minus catR's `alpha`. The parity tests
({doc}`ADR-0020 </adr/0020-catr-as-the-reference>`) find catR's decisions with
the interval above, across and below the threshold, at `alpha` of 0.05, 0.1 and
0.01, before any response, and with the interval's end $10^{-9}$ either side of
the threshold. catR's `stop` list can hold several classification rules, and
stops once the interval clears any one threshold: that is a
{class}`~catsim.stopping.StopRule` of one-boundary stoppers, such as
`StopRule(ConfidenceIntervalStopper([-1.0]), ConfidenceIntervalStopper([1.0]))`,
whose decisions the tests also find, and not one stopper with both boundaries,
which waits until the interval lies within one band. No whole test compares the
rule.

## Guarantees

1. The stopper stops the test exactly when the full confidence interval lies within a single induced classification interval.
2. If the confidence interval crosses any configured boundary, the test does not stop.
3. Before any response the standard error is infinite, so the test does not stop.

## References

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. autoclass:: catsim.stopping.ConfidenceIntervalStopper
   :members:
   :show-inheritance:
```
