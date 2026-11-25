# ADR-0009: Explicit Inputs Are Checked Where They Enter

## Status

Accepted 2026-11-25

## Context

With steps computed from explicit inputs (ADR-0006), an application passes the items given, the
responses, the estimate and the exposure rates at every call, from its own storage. A computation on
a bad input rarely fails: NumPy computes on NaN, an index of 3.7 reads item 3, and a repeated item
counts twice. The result is a plausible next item or estimate that is wrong, and nobody notices.

The step context already rejected responses that are not Booleans, responses that do not match the
items in length, and item indices out of range. Fed eleven bad inputs, it accepted five silently:

- a repeated item, which changed the estimate;
- an item index of 3.7, read as item 3;
- a NaN estimate, which changed the selected item;
- a standard error of -1;
- exposure counts passed as rates, which turned the exposure cap off.

Exposure rates of the wrong length failed later, deep in NumPy. catR's `nextItem` checks only the
names of its criterion and model, and `thetaEst` silently drops missing responses.

## Decision

- Every public computation checks its explicit inputs where they enter catsim, when a `Session`,
  `StepContext` or `ItemBank` is built and when a step method is called. It checks everything that
  can be checked without recomputation:
  - item indices are integers, in range, with no repeats, and no item is both administered and
    excluded;
  - there is one response per administered item, each a Boolean or the integer 0 or 1;
  - estimates are finite, and standard errors are positive, possibly infinite: an estimator returns
    an infinite error before any response and after responses that carry no information;
  - exposure rates hold one value per item, each in $[0, 1]$.
- A failed check raises `TypeError` or `ValueError`, naming the input, the position, the value and
  the rule: "administered_items[1] is 3.7; item indices must be integers". Nothing is converted
  silently.
- catsim does not check what only recomputation would reveal, such as whether an estimate is the
  one the responses give or whether the exposure rates are the application's real ones. Those are
  the caller's.
- Strategies receive checked inputs, and the simulation runner may skip the checks for the inputs
  it builds itself.

## Consequences

- A storage or format mistake in an application fails at the call that receives it, with a message
  that points to it.
- Method authors do not repeat the checks in every strategy.
- The boundary page tells integrators what catsim checks and what stays theirs.
- An application that stores answers as 0 and 1 passes them as they are.
- Checking costs a few microseconds per call.

## Alternatives Considered

- Checking nothing, as catR: the five silent cases above stay silent.
- Checking consistency by recomputing the estimate from the responses: it doubles the cost of a step,
  and an application may pass an estimate from another estimator on purpose.
- Converting what can be converted, as `bool("0")`: `bool` of any non-empty string is `True`.
- Checks inside each strategy: every method author would repeat them, and some would not.
