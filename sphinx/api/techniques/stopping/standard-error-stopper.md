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

Let $I(\hat{\theta})$ be the information of the administered test at the
current estimate. The standard error of estimation is

$$
SEE(\hat{\theta})
=
\sqrt{\frac{1}{I(\hat{\theta})}}.
$$

The stopper stops when

$$
SEE(\hat{\theta}) \le \tau,
$$

where $\tau$ is the configured `max_se`. With no administered items the
information is zero and the standard error infinite, so the test does not stop.
As in catR, items are administered "until the standard error of the provisional
ability estimate gets smaller than or equal to that threshold" (Magis and Raîche,
2012, p. 11).

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

1. The stopper returns `True` exactly when the standard error at the current estimate is at most `max_se`.
2. If no items have been administered, the stopper returns `False`.
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
