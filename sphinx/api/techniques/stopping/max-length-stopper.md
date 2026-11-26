# Maximum Length Stopper

:Class: {py:class}`catsim.stopping.MaxLengthStopper`
:Sources: {cite:t}`Wainer2000`; {cite:t}`Thissen00`

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

## Behavior and Comparison

It does not look at the item bank or at the ability estimate. A session whose
bank runs out before $m_{\max}$ items is ended by the engine, once the selector
has no item left to give.

### Comparison with catR

The stopper is catR's `length` rule. The parity tests
({doc}`ADR-0020 </adr/0020-catr-as-the-reference>`) find catR's decisions one
item below, at and above the length, alone and with a precision rule, and catR's
test lengths in whole tests on recorded responses. They find no departure.

## Guarantees

1. The stopper stops the test exactly when at least `max_items` items have been administered.
2. It ignores the item bank and the ability estimate; in particular, administering the whole bank does not stop the test.

## References

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. autoclass:: catsim.stopping.MaxLengthStopper
   :members:
   :show-inheritance:
```
