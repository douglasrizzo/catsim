# Initialization Techniques

```{py:module} catsim.initialization
```

Initialization techniques choose the initial ability estimate
$\hat{\theta}_0$ before an adaptive test has administered any items. The
choice of initializer affects the first item selection and can be useful for
controlled simulations, reproducible baselines, or stochastic starting points.

{mod}`catsim` has three initializers:
{py:class}`~catsim.initialization.FixedPointInitializer` starts every examinee
at the same value, and {py:class}`~catsim.initialization.UniformInitializer`
and {py:class}`~catsim.initialization.NormalInitializer` draw the value from a
distribution. Every initializer receives the item bank to satisfy the
initializer contract; none of these three reads it.

## Fixed Point Initializer

Fixed-point initialization gives every examinee the same initial ability
estimate. It is useful for deterministic baselines, controlled simulation
studies, and workflows that should not introduce randomness before the first
item is selected.

### Definition

For a configured starting value $\theta_0$, the initializer returns the
same value for every examinee:

$$
\hat{\theta}_0 = \theta_0.
$$

### Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``start``
     - Required
     - Ability estimate returned for every examinee.
```

### Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. It rejects a `start` that is not a finite real number when it is constructed.
2. It returns the configured `start` value, as a `float`, for every initialization request.
3. It does not consume random numbers from the provided generator.

### API Reference

```{eval-rst}
.. autoclass:: catsim.initialization.FixedPointInitializer
   :members:
   :show-inheritance:
```

## Random Initializers

Random initialization samples the initial ability estimate from a configured
distribution. It is useful when simulations should vary the starting point
across examinees while remaining reproducible through the engine or runner
random number generator.

### Definition

{py:class}`~catsim.initialization.UniformInitializer` samples

$$
\hat{\theta}_0 \sim U(\text{low}, \text{high}).
$$

{py:class}`~catsim.initialization.NormalInitializer` samples

$$
\hat{\theta}_0 \sim \mathcal{N}(\text{mean}, \text{sd}^2).
$$

The default uniform range, $(-0.5, 0.5)$, is the starting rule of Barrada et
al. (2006, p. 158) and Barrada et al. (2008, p. 499), who drew the first
estimate from it for examinees simulated from $\mathcal{N}(0, 1)$. The normal
default, $\mathcal{N}(0, 1)$, is not taken from a source; it matches the
ability distribution those studies simulate.

### Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
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

### Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. It rejects non-finite parameters when it is constructed.
2. A uniform initializer rejects `high <= low` and samples between `low` and `high`.
3. A normal initializer rejects a non-positive `sd`.
4. It samples using the random number generator passed to `initialize`.

### API Reference

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

## References

Barrada, J. R., Mazuela, P., & Olea, J. (2006). Maximum information stratification method for controlling item exposure in computerized adaptive testing. *Psicothema*, 18(1), 156-159. <https://www.psicothema.com/pdf/3191.pdf> (open access).

Barrada, J. R., Olea, J., Ponsoda, V., & Abad, F. J. (2008). Incorporating randomness in the Fisher information for improving item-exposure control in CATs. *British Journal of Mathematical and Statistical Psychology*, 61(2), 493-513. <https://doi.org/10.1348/000711007X230937>. Open copy: [first author's page](https://personal.unizar.es/barrada/papers/08_BJMSP_Barrada.pdf).

## Class Hierarchy

```{autoclasstree} catsim.initialization
:full:
:strict:
:namespace: catsim
```

## Base Class

```{eval-rst}
.. autoclass:: catsim.initialization.BaseInitializer
   :members:
```
