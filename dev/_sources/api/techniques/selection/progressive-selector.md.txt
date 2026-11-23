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

At test position $n$ out of a planned total length $N$, the candidate
score for item $i$ is

$$
S_i^{(n)}
=
(1-w_n)\,R_i
+
w_n\,\frac{I_i(\hat{\theta})}{\max_{j \in \mathcal{A}} I_j(\hat{\theta})},
$$

where $\mathcal{A}$ is the set of available items (the non-administered items
under the exposure cap, or every non-administered item when the cap excludes
them all), $R_i$ is a fresh uniform random draw on $[0,1]$, and the weight on
information is

$$
w_n
=
\frac{\sum_{m=1}^{n-1} m^{t}}{\sum_{m=1}^{N-1} m^{t}}
\quad (n < N),
\qquad
w_n = 1
\quad (n \ge N),
$$

with acceleration parameter $t$, any real number. The weight is 0 at the first
position and 1 at the last. With $t = 0$ it is linear, $w_n = (n-1)/(N-1)$;
positive $t$ keeps the random component longer and negative $t$ drops it sooner
(Barrada et al., 2008, p. 498; Figure 1, p. 499, plots $t = -1, 0, 1, 2$).
Dividing by the largest available information is equivalent to Revuelta and
Ponsoda's (1998) random component drawn on $[0, \max_{j \in \mathcal{A}} I_j]$,
as Barrada et al. (2008, p. 496, Equation 6) describe it. The selector chooses
the available item with the largest score.

The schedule is Barrada et al.'s (2008, p. 498, Equation 12), with $m$ for their
$h - 1$ and $n$, $N$ for their $k$, $K$. As printed, both sums in Equation 12 run
to $k$, and Equations 12 and 13 both use the exponent $s$; the text under them,
which names $t$ and $s$ as the two acceleration parameters and gives a weight of
1 at the last item, puts $K$ in the denominator and $t$ in Equation 12. `catR`
computes it that way in
[`nextItem.R`](https://github.com/cran/catR/blob/master/R/nextItem.R), with its
argument `AP` as $t$ and `thr` as $N$ when `rule = "length"`. catsim departs
from both in three places:

- The default is $t = 0$, the linear schedule; `catR`'s `nextItem` defaults to
  `AP = 1`.
- Past position $N$, which a session reaches when its stopper allows more items
  than planned, the weight stays 1. In `catR` the sums keep growing, and the
  weight passes 1 and gives the random component a negative coefficient.
- With $N = 1$ the only position is the last, and the weight is 1. Equation 12
  and `catR` give 0 at the first position whatever $N$ is.

`catR`'s other schedule, for its precision stopping rule, is not implemented.
Revuelta and Ponsoda's original schedule, $w = (n - 1)/N$, stops just short of 1
at the last item (Barrada et al., 2008, p. 496, Equation 7); Barrada et al. did
not test it separately, since it differs little from Equation 12 with $t = 0$
(p. 500).

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
     - Planned test length :math:`N`, at least 1; required because the schedule depends on test position.
   * - ``acceleration``
     - ``0.0``
     - Acceleration parameter :math:`t`, any finite number. 0 makes the schedule linear; positive values keep the random component longer and negative values drop it sooner.
   * - ``r_max``
     - ``1.0``
     - Maximum exposure rate, in :math:`(0, 1]`. Items at or above it are left
       out before the scores are computed, unless every remaining item is.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. The blend weight is exactly 0 at position 1 and exactly 1 at the final test position when `test_size > 1`.
2. With `acceleration = 0`, the schedule is linear in test position; positive acceleration delays the shift toward information-based scoring and negative acceleration speeds it up.
3. At the first position, selection is driven only by the random component and is therefore independent of item information.
4. At the final position, the score reduces to normalized information and behaves like an information-based selector subject to the exposure rule.
5. The random component must be drawn from the provided RNG when one is supplied so repeated simulations remain reproducible.

## References

Revuelta, J., & Ponsoda, V. (1998). A comparison of item exposure control methods in computerized adaptive testing. *Journal of Educational Measurement*, 35(4), 311-327. <https://doi.org/10.1111/j.1745-3984.1998.tb00541.x>. Not openly available; Barrada et al. (2008) describe the method.

Barrada, J. R., Olea, J., Ponsoda, V., & Abad, F. J. (2008). Incorporating randomness in the Fisher information for improving item-exposure control in CATs. *British Journal of Mathematical and Statistical Psychology*, 61(2), 493-513. <https://doi.org/10.1348/000711007X230937>. Open copy: [first author's page](https://personal.unizar.es/barrada/papers/08_BJMSP_Barrada.pdf).

Barrada, J. R., Olea, J., Ponsoda, V., & Abad, F. J. (2010). A method for the comparison of item selection rules in computerized adaptive testing. *Applied Psychological Measurement*, 34(6), 438-452. <https://doi.org/10.1177/0146621610370152>. Open copy: [first author's page](https://personal.unizar.es/barrada/papers/10_APM_Barrada.pdf).

:::{seealso}
Companion spec: {doc}`/api/techniques/selection/proportional-selector`
:::

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.ProgressiveSelector
   :members:
   :show-inheritance:
```
