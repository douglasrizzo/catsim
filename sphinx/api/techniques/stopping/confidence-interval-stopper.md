**Status:** Implemented
**Module:** {py:class}`catsim.stopping.ConfidenceIntervalStopper`
**Reference:** Thissen (2000); Wainer (2000)

# Confidence Interval Stopper

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
  \hat{\theta} - z_{(1+\gamma)/2} \, SEE(\hat{\theta}),
  \hat{\theta} + z_{(1+\gamma)/2} \, SEE(\hat{\theta})
\right],
$$

where $\gamma$ is the configured confidence level and
$SEE(\hat{\theta}) = \sqrt{1 / I(\hat{\theta})}$.

Given sorted boundary points $b_1 < \dots < b_m$, the rule stops when the
entire confidence interval falls within one interval induced by those boundaries:

$$
(-\infty, b_1],\ [b_1, b_2],\ \dots,\ [b_m, \infty).
$$

Equivalently, the interval lies within one band exactly when no boundary lies
strictly inside it, which is what the implementation checks. With no administered
items the interval is infinite, so the test does not stop.

Magis and Raîche (2012, p. 11) describe the classification criterion with a
single ability threshold: a confidence interval for ability is built after every
item, and the test stops once the threshold is no longer inside it. catsim
generalizes this to several boundaries; with one boundary the two rules agree.

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

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. `interval_bounds` must be nonempty and sorted in ascending order, and `confidence` must lie strictly between 0 and 1; otherwise the constructor raises `ValueError`.
2. The stopper returns `True` exactly when the full confidence interval lies within a single induced classification interval.
3. If the confidence interval crosses any configured boundary, the stopper returns `False`.
4. If no items have been administered, the stopper returns `False`.

## References

Thissen, D. (2000). Reliability and measurement precision. In H. Wainer (Ed.), *Computerized Adaptive Testing: A Primer* (2nd ed., pp. 159-184). Mahwah, NJ: Lawrence Erlbaum Associates.

Wainer, H. (Ed.). (2000). *Computerized Adaptive Testing: A Primer* (2nd ed.). Mahwah, NJ: Lawrence Erlbaum Associates.

Magis, D., & Raîche, G. (2012). Random generation of response patterns under computerized adaptive testing with the R package catR. *Journal of Statistical Software*, 48(8), 1-31. <https://doi.org/10.18637/jss.v048.i08> (open access).

## API Reference

```{eval-rst}
.. autoclass:: catsim.stopping.ConfidenceIntervalStopper
   :members:
   :show-inheritance:
```
