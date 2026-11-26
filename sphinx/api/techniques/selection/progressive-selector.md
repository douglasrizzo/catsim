# Progressive Selector

:Class: {py:class}`catsim.selection.ProgressiveSelector`
:Sources: {cite:t}`Revuelta1998`; {cite:t}`Barrada2008,Bar10`

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

where $\mathcal{A}$ is the set of available items (the remaining items, neither
administered nor excluded, under the exposure cap, or every remaining item when
the cap excludes them all), $R_i$ is a fresh uniform random draw on $[0,1]$, and the weight on
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
{cite:p}`Barrada2008{p. 498}` (Figure 1, p. 499, plots $t = -1, 0, 1, 2$).
Dividing by the largest available information is equivalent to
{cite:author}`Revuelta1998`'s {cite:yearpar}`Revuelta1998` random component drawn on
$[0, \max_{j \in \mathcal{A}} I_j]$, as {cite:t}`Barrada2008{p. 496, Equation 6}` describe
it. The selector chooses the available item with the largest score.

The schedule is that of {cite:t}`Barrada2008{p. 498, Equation 12}`, with $m$ for their
$h - 1$ and $n$, $N$ for their $k$, $K$. As printed, both sums in Equation 12 run
to $k$, and Equations 12 and 13 both use the exponent $s$; the text under them,
which names $t$ and $s$ as the two acceleration parameters and gives a weight of
1 at the last item, puts $K$ in the denominator and $t$ in Equation 12.

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

## Behavior and Comparison

Revuelta and Ponsoda's original schedule, $w = (n - 1)/N$, stops just short of 1
at the last item {cite:p}`Barrada2008{p. 496, Equation 7}`; Barrada et al. did
not test it separately, since it differs little from Equation 12 with $t = 0$
(p. 500).

### Comparison with catR

`catR` computes the schedule as above in
[`nextItem.R`](https://github.com/cran/catR/blob/master/R/nextItem.R), with its
argument `AP` as $t$ and `thr` as $N$ when `rule = "length"`
({doc}`ADR-0019 </adr/0019-published-progressive-schedule>`). Within the planned
length, the parity tests ({doc}`ADR-0020 </adr/0020-catr-as-the-reference>`)
find catR's weight and, given catR's random draws, its choice at every position
of 5-item tests with $t = -0.5$, 0, 1 and 2.5 and of an 8-item test with
$t = 1$; and a simulation of 1,000 examinees taking 10-item tests with $t = 1$
gives catR's bias, RMSE, overlap rate and exposure rates within their Monte
Carlo error. catsim departs from catR in three places, the last two from
Equation 12 as well:

- The default is $t = 0$, the linear schedule, where catR's `nextItem` defaults
  to `AP = 1`; `acceleration=1` gives catR's schedule.
- Past position $N$, which a session reaches when its stopper allows more items
  than planned, the weight stays 1, so the selector keeps choosing by
  information. In catR the sums keep growing, and the weight passes 1, which
  gives the random component a negative coefficient.
- With $N = 1$ the only position is the last, and the weight is 1. Equation 12
  and catR give 0 at the first position whatever $N$ is, which makes a one-item
  test random.

The parity tests keep the last two as expected failures. catR's other schedule,
for its precision stopping rule, has no catsim counterpart.

## Guarantees

1. It never returns an item that was already administered or excluded.
2. When `test_size` is above 1, the choice at the first position does not depend on item
   information: every available item is equally likely.
3. From the `test_size`-th item on, it returns the most informative available item.
4. When every remaining item is at or above the exposure cap, it chooses among all
   the remaining items.
5. It draws with the session's random generator, so a run with the same seed
   administers the same items.

## References

```{bibliography}
:filter: docname in docnames
```

:::{seealso}
{doc}`/api/techniques/selection/proportional-selector`, which follows the same schedule
:::

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.ProgressiveSelector
   :members:
   :show-inheritance:
```
