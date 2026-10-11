**Status:** Implemented
**Module:** {py:class}`catsim.selection.ProgressiveSelector`
**Reference:** Revuelta & Ponsoda (1998); Barrada et al. (2008, 2010)

# Progressive Selector

## Motivation

Pure maximum-information selection tends to overuse a small set of highly
informative items. Progressive selection addresses that exposure imbalance by
starting almost random and gradually moving toward information-driven scoring as
the test advances.

This gives the method a clear operational role: better exposure control early,
near-Fisher behavior late, and no need for offline calibration.

## Definition

At test position $n$ out of a fixed total length $N$, the candidate
score for item $i$ is

$$
S_i^{(n)}
=
(1-w_n)\,R_i
+
w_n\,\frac{I_i(\hat{\theta})}{\max_{j \in \mathcal{A}} I_j(\hat{\theta})},
$$

where $\mathcal{A}$ is the set of available (non-administered) items,
$R_i$ is a fresh uniform random draw on $[0,1]$, and

$$
w_n
=
\min\!\left(1, \frac{n-1}{N-1}\right)^s,
$$

with acceleration parameter $s > 0$. Dividing by the largest available
information is equivalent to Revuelta and Ponsoda's (1998) random component
drawn on $[0, \max_{j \in \mathcal{A}} I_j]$. The weight is clamped at 1 so
that a session running past $N$ stays fully information-based. The selector
chooses the available item with the largest score.

Barrada et al. (2008) parameterize the schedule as a cumulative power sum with
an acceleration parameter $t$,
$w = \sum_{k=1}^{n-1} k^{t} / \sum_{k=1}^{N-1} k^{t}$ (as implemented in
`catR`). It agrees with the power schedule above at $t = 0$, $s = 1$ (linear),
and behaves like $s \approx t + 1$ otherwise, so `acceleration` is not
interchangeable with their $t$. Revuelta and Ponsoda's original schedule,
$w = (n - 1)/N$, stops just short of 1 at the last item.

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
     - Total number of administered items :math:`N`; required because the schedule depends on test position.
   * - ``acceleration``
     - ``1.0``
     - Positive exponent :math:`s`. Values above 1 delay the transition to information-based selection; values below 1 accelerate it.
   * - ``r_max``
     - ``1.0``
     - Exposure cap applied after scoring candidates.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. The blend weight is exactly 0 at position 1 and exactly 1 at the final test position when `test_size > 1`.
2. With `acceleration = 1`, the schedule is linear in test position; larger acceleration delays the shift toward information-based scoring and smaller acceleration speeds it up.
3. At the first position, selection is driven only by the random component and is therefore independent of item information.
4. At the final position, the score reduces to normalized information and behaves like an information-based selector subject to the exposure rule.
5. The random component must be drawn from the provided RNG when one is supplied so repeated simulations remain reproducible.

## References

Revuelta, J., & Ponsoda, V. (1998). A comparison of item exposure control methods in computerized adaptive testing. *Journal of Educational Measurement*, 35(4), 311-327.

Barrada, J. R., Olea, J., Ponsoda, V., & Abad, F. J. (2008). Incorporating randomness in the Fisher information for improving item-exposure control in CATs. *British Journal of Mathematical and Statistical Psychology*, 61(2), 493-513. <https://doi.org/10.1348/000711007X230937>

Barrada, J. R., Olea, J., Ponsoda, V., & Abad, F. J. (2010). A method for the comparison of item selection rules in computerized adaptive testing. *Applied Psychological Measurement*, 34(6), 438-452.

:::{seealso}
Companion spec: {doc}`/api/techniques/selection/proportional-selector`
:::

## Reading the Source

Revuelta and Ponsoda (1998) was **not checked**: it is not openly available.
The method was checked against its description in the full text of Barrada et
al. (2008), [open copy on the first author's page](https://personal.unizar.es/barrada/papers/08_BJMSP_Barrada.pdf):

- **p. 496, Equation 6:** the progressive method selects
  $\max_i\, [(1 - W_k) R_i + W_k I_i(\hat{\theta})]$, where $R_i$ is a random
  number in $[0, \max_i I_i(\hat{\theta})]$. Dividing by that maximum, as
  catsim does, puts both terms on $[0, 1]$ without changing the selection.
- **p. 496, Equation 7:** Revuelta and Ponsoda's linear weight, rising from 0
  to $(K - 1)/K$ for a $K$-item test.
- **p. 498, Equation 12:** the cumulative power-sum weight, which is 0 at the
  first item and 1 at the last, with an acceleration parameter: 0 gives a
  linear schedule, and positive values keep the random component longer.
  **Figure 1** (p. 499) plots it.
- **p. 499:** their simulation design (500-item banks, 25- and 40-item tests).

Barrada et al. (2010) compares this and other selection rules across exposure
caps; it was not checked for this page.

**Checked by** `TestProgressiveSelector` in `tests/test_selection.py`, including
`test_weight_is_clamped_past_test_size` and
`test_random_component_is_scaled_to_available_items`.

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.ProgressiveSelector
   :members:
   :show-inheritance:
```
