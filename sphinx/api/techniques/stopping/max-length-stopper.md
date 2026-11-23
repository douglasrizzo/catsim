**Status:** Implemented
**Module:** {py:class}`catsim.stopping.MaxLengthStopper`
**Reference:** Wainer (2000); Thissen (2000)

# Maximum Length Stopper

## Motivation

Fixed-length stopping is the simplest CAT termination rule: the test ends after a
predefined number of administered items. Variable-length tests need the same
limit as a cap, because a precision or classification target can be out of reach
for some examinees. In {mod}`catsim` the cap is this stopper, combined with the
other criterion in a {class}`~catsim.stopping.StopRule` (see
{ref}`combining-stopping-rules`).

## Definition

Let $m$ be the number of administered items. The stopper stops when

$$
m \ge m_{\max}.
$$

It does not look at the item bank or at the ability estimate. A session whose
bank runs out before $m_{\max}$ items is ended by the engine, once the selector
has no item left to give.

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``max_items``
     - required
     - Number of administered items at which the test stops. Must be at least 1.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. The stopper returns `True` exactly when at least `max_items` items have been administered.
2. It ignores the item bank and the ability estimate; in particular, administering the whole bank does not stop the test.
3. A `max_items` below 1 raises `ValueError`.

## References

Wainer, H. (Ed.). (2000). *Computerized Adaptive Testing: A Primer* (2nd ed.). Mahwah, NJ: Lawrence Erlbaum Associates.

Thissen, D. (2000). Reliability and measurement precision. In H. Wainer (Ed.), *Computerized Adaptive Testing: A Primer* (2nd ed., pp. 159-184). Mahwah, NJ: Lawrence Erlbaum Associates.

## API Reference

```{eval-rst}
.. autoclass:: catsim.stopping.MaxLengthStopper
   :members:
   :show-inheritance:
```
