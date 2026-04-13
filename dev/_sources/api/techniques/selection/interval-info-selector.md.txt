**Status:** Implemented
**Module:** {py:class}`catsim.selection.IntervalInfoSelector`
**Reference:** Veerkamp & Berger (1997)

# Interval Information Selector

## Motivation

Point-information rules assume the current estimate is accurate enough to be
trusted as a single design point. The interval-information criterion relaxes
that assumption by averaging each item's information over a symmetric interval
around the current estimate, which makes the rule less sensitive to early-stage
estimation noise.

This selector is useful when a user wants a deterministic alternative to
maximum-information selection that still remains close to the standard IRT
information framework.

## Definition

For interval half-width $\delta$, the selector scores each candidate item by

$$
S_i(\hat{\theta}, \delta)
=
\int_{\hat{\theta}-\delta}^{\hat{\theta}+\delta} I_i(\theta)\,d\theta.
$$

The selected item is

$$
i^*
=
\arg\max_{i \in R_k} S_i(\hat{\theta}, \delta).
$$

In this package, `interval=None` is implemented as $\delta = \infty$,
so the integral is taken over the full real line using SciPy quadrature.

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
     - Symmetric half-width :math:`\delta` around ``est_theta``. When ``None``,
       the implementation integrates over :math:`(-\infty, \infty)`.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. It computes each candidate score as a symmetric integral of the item information function around `est_theta`.
2. It never returns an already administered item.
3. With a very small interval, the ranking approaches that of local maximum-information selection.
4. With `interval=None`, the score is the total integrated information over the full ability axis.
5. If no non-administered items remain, it raises `NoItemsAvailableError`.

## References

Veerkamp, W. J. J., & Berger, M. P. F. (1997). Some new item selection criteria for adaptive testing. *Journal of Educational and Behavioral Statistics*, 22(2), 203-226. <https://doi.org/10.3102/10769986022002203>

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.IntervalInfoSelector
   :members:
   :show-inheritance:
```
