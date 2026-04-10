**Status:** Implemented
**Module:** {py:class}`catsim.selection.ProportionalSelector`
**Reference:** Segall (2004); Barrada et al. (2008)

# Proportional Selector

## Motivation

The proportional selector uses the same position-dependent schedule as the
progressive selector, but instead of blending random and information scores, it
samples directly from a probability distribution defined by item information.

That gives the method a clean probabilistic interpretation: fully random at the
start of the test, increasingly concentrated on high-information items as the
test progresses, and asymptotically close to deterministic Fisher selection.

## Definition

At position $n$, item $i$ is selected with probability

$$
\Pr(\text{select } i)
=
\frac{I_i(\hat{\theta})^{h(w_n)}}
     {\sum_j I_j(\hat{\theta})^{h(w_n)}},
$$

where the exponent schedule is

$$
h(w_n) = k\,w_n,
\qquad
w_n = \left(\frac{n-1}{N-1}\right)^s.
$$

Here $k > 0$ is a sharpness parameter and $s > 0$ is the same
acceleration parameter used by the progressive selector.

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Parameter
     - Default
     - Description
   * - ``test_size``
     - required
     - Fixed test length :math:`N`.
   * - ``acceleration``
     - ``1.0``
     - Positive schedule exponent :math:`s`.
   * - ``sharpness``
     - ``6.0``
     - Positive constant :math:`k` controlling how quickly the sampling distribution concentrates on high-information items.
   * - ``r_max``
     - ``1.0``
     - Exposure cap applied before sampling; if every remaining candidate exceeds the cap, the selector falls back to ignoring the cap.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. At the first test position, the exponent is 0 and the sampling distribution is uniform over eligible candidates regardless of the sharpness value.
2. Increasing sharpness at late test positions makes the distribution more concentrated on the highest-information items.
3. When all candidate information values are nonpositive, the selector falls back to uniform sampling rather than producing invalid probabilities.
4. Exposure filtering is applied before sampling, with fallback to the unconstrained non-administered pool only when every constrained candidate is excluded.
5. The selector uses the provided RNG for sampling so its stochastic behavior can be reproduced in tests and simulations.

## References

Segall, D. O. (2004). Computerized adaptive testing. In K. Kempf-Leonard (Ed.), *Encyclopedia of Social Measurement*. Academic Press.

Barrada, J. R., Olea, J., Ponsoda, V., & Abad, F. J. (2008). Incorporating randomness in the Fisher information function for item exposure control. *Methodology*, 4(2), 51-59.

:::{seealso}
Companion spec: {doc}`/api/techniques/selection/progressive-selector`
:::
