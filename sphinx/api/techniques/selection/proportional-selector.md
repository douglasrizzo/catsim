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
w_n = \min\!\left(1, \frac{n-1}{N-1}\right)^s.
$$

Here $k > 0$ is a sharpness parameter and $s > 0$ is the same
acceleration parameter used by the progressive selector; the sums run over
the available items, and the weight is clamped at 1 past position $N$.

In Barrada et al. (2008), as implemented in `catR`, the exponent grows to the
test length, $h = N\,w$ with the cumulative power-sum schedule described for
the progressive selector. Setting `sharpness` to the test length reproduces
that end point; the default $k = 6$ keeps more randomness late in longer tests.

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

Segall, D. O. (2004). A sharing item response theory model for computerized adaptive testing. *Journal of Educational and Behavioral Statistics*, 29(4), 439-460.

Barrada, J. R., Olea, J., Ponsoda, V., & Abad, F. J. (2008). Incorporating randomness in the Fisher information for improving item-exposure control in CATs. *British Journal of Mathematical and Statistical Psychology*, 61(2), 493-513. <https://doi.org/10.1348/000711007X230937>

:::{seealso}
Companion spec: {doc}`/api/techniques/selection/progressive-selector`
:::

## Reading the Source

Segall (2004), where the proportional idea originates, was **not checked**: it
is not openly available. The method was checked against the full text of
Barrada et al. (2008), [open copy on the first author's page](https://personal.unizar.es/barrada/papers/08_BJMSP_Barrada.pdf):

- **p. 497, Equations 8–10:** Segall's proportional method selects item $i$
  with probability $V_i / \sum_j V_j$, by drawing from the cumulative
  distribution. In Segall's paper $V_i$ was an expected variance; with Fisher
  information, $V_i = I_i(\hat{\theta})$.
- **p. 497, Equation 11:** Barrada et al.'s modification raises the information
  to a power that grows during the test, $V_i = I_i(\hat{\theta})^{P_k}$. A power
  of 0 gives random selection and a large power approaches maximum information.
- **p. 498, Equation 13:** the power is $P_k = K W_k$, with the same cumulative
  power-sum schedule as the progressive method, so it reaches the test length
  $K$ at the last item.

catsim's exponent is $k\,w_n$ with a fixed sharpness $k$ (6 by default) and a
power schedule for $w_n$; setting `sharpness` to the test length reproduces
Barrada et al.'s end point.

**Checked by** `TestProportionalSelector` in `tests/test_selection.py`, including
`test_first_position_is_uniform_regardless_of_sharpness` and
`test_exponent_is_clamped_past_test_size`.

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.ProportionalSelector
   :members:
   :show-inheritance:
```
