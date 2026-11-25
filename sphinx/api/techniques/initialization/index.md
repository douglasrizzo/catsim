# Initialization Techniques

:Class: {py:class}`catsim.initialization.FixedPointInitializer`
:Class: {py:class}`catsim.initialization.UniformInitializer`
:Class: {py:class}`catsim.initialization.NormalInitializer`
:Sources: {cite:t}`Barrada2006`; {cite:t}`Barrada2008`

```{py:module} catsim.initialization
```

## Motivation

Initialization techniques choose the initial ability estimate
$\hat{\theta}_0$ before an adaptive test has administered any items. The
choice of initializer affects the first item selection and can be useful for
controlled simulations, reproducible baselines, or stochastic starting points.

{mod}`catsim` has three initializers:
{py:class}`~catsim.initialization.FixedPointInitializer` starts every examinee
at the same value, and {py:class}`~catsim.initialization.UniformInitializer`
and {py:class}`~catsim.initialization.NormalInitializer` draw the value from a
distribution.

Fixed-point initialization is useful for deterministic baselines, controlled
simulation studies, and workflows that should not introduce randomness before
the first item is selected. Random initialization is useful when simulations
should vary the starting point across examinees while remaining reproducible
through the engine or runner random number generator.

## Definition

For a configured starting value $\theta_0$,
{py:class}`~catsim.initialization.FixedPointInitializer` returns the same value
for every examinee:

$$
\hat{\theta}_0 = \theta_0.
$$

{py:class}`~catsim.initialization.UniformInitializer` samples

$$
\hat{\theta}_0 \sim U(\text{low}, \text{high}).
$$

{py:class}`~catsim.initialization.NormalInitializer` samples

$$
\hat{\theta}_0 \sim \mathcal{N}(\text{mean}, \text{sd}^2).
$$

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``FixedPointInitializer(start)``
     - Required
     - Ability estimate returned for every examinee.
   * - ``UniformInitializer(low)``
     - ``-0.5``
     - Lower bound of the interval.
   * - ``UniformInitializer(high)``
     - ``0.5``
     - Upper bound of the interval; must be greater than ``low``.
   * - ``NormalInitializer(mean)``
     - ``0.0``
     - Mean of the distribution.
   * - ``NormalInitializer(sd)``
     - ``1.0``
     - Standard deviation of the distribution; must be positive.
```

The default uniform range, $(-0.5, 0.5)$, is the starting rule of
{cite:t}`Barrada2006{p. 158}` and {cite:t}`Barrada2008{p. 499}`, who drew the first
estimate from it for examinees simulated from $\mathcal{N}(0, 1)$. The normal
default, $\mathcal{N}(0, 1)$, is not taken from a source; it matches the
ability distribution those studies simulate.

## Guarantees

1. A fixed-point initializer starts every examinee at `start`, as a `float`,
   whatever the seed.
2. A uniform initializer's draws lie between `low` and `high`.
3. The random initializers draw from the run's random generator, so the same
   seed gives the same initial estimates.

## References

```{bibliography}
:filter: docname in docnames
```

## API Reference

```{eval-rst}
.. autoclass:: catsim.initialization.FixedPointInitializer
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: catsim.initialization.UniformInitializer
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: catsim.initialization.NormalInitializer
   :members:
   :show-inheritance:
```

### Class Hierarchy

```{autoclasstree} catsim.initialization
:full:
:strict:
:namespace: catsim
```

### Base Class

```{eval-rst}
.. autoclass:: catsim.initialization.BaseInitializer
   :members:
```
