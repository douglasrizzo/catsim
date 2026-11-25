# Proportional Selector

:Class: {py:class}`catsim.selection.ProportionalSelector`
:Sources: {cite:t}`Segall2004`; {cite:t}`Barrada2008`

## Motivation

The proportional selector uses the same position-dependent schedule as the
progressive selector, but rather than blend random and information scores, it
samples directly from a probability distribution defined by item information.

That gives the method a clean probabilistic interpretation: fully random at the
start of the test and increasingly concentrated on high-information items as the
test progresses. At the last item the information is raised to the test length,
so the longer the test, the closer its last draws come to maximum-information
selection; for the test lengths CATs use, {cite:t}`Barrada2008{p. 498}` expect
items of low information to be drawn with negligible probability by then.

## Definition

At position $n$ of a test of planned length $N$, item $i$ is selected with
probability

$$
\Pr(\text{select } i)
=
\frac{I_i(\hat{\theta})^{h_n}}
     {\sum_{j \in \mathcal{A}} I_j(\hat{\theta})^{h_n}},
\qquad
h_n = N\,w_n,
$$

where $\mathcal{A}$ is the set of available items (the remaining items under the
exposure cap, or every remaining item when the cap excludes them all) and $w_n$
is the progressive selector's schedule with its own acceleration parameter $s$,
any real number:

$$
w_n
=
\frac{\sum_{m=1}^{n-1} m^{s}}{\sum_{m=1}^{N-1} m^{s}}
\quad (n < N),
\qquad
w_n = 1
\quad (n \ge N).
$$

The exponent is 0 at the first position, where selection is uniformly random,
and grows to the test length $N$ at the last; with $s = 0$ it grows linearly.
Past position $N$ it stays at $N$.

{cite:t}`Segall2004` selects each item with probability proportional to $V_i$; in
his paper $V_i$ was an expected variance, and with Fisher information
$V_i = I_i(\hat{\theta})$ {cite:p}`Barrada2008{p. 497, Equations 8–10}`.
Barrada et al. raise the information to a power that grows during the test
(p. 497, Equation 11): a power of 0 gives random selection, a power of 1
Segall's method, and a large power approaches maximum information
(pp. 497–498). Their power is $h_n$ above (p. 498, Equation 13, read with $K$ in
the denominator as the progressive selector's page explains for Equation 12).

### Computation

`catR` computes the power the same way in
[`nextItem.R`](https://github.com/cran/catR/blob/master/R/nextItem.R), as `thr`
times the progressive weight. catsim departs from both where the progressive
selector does: the default is $s = 0$, where `catR`'s `AP` defaults to 1; past
position $N$ the exponent stays at $N$, where `catR`'s keeps growing; and with
$N = 1$ the exponent is 1, Segall's method, where Equation 13 and `catR` give 0.

catsim divides the information by its maximum over the available items before
raising it to the power, which leaves the probabilities unchanged and keeps a
large exponent from underflowing every weight to zero.

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
     - Planned test length :math:`N`, at least 1, which the exponent reaches at the last position.
   * - ``acceleration``
     - ``0.0``
     - Acceleration parameter :math:`s`, any finite number. 0 makes the exponent grow linearly; positive values keep sampling near uniform longer and negative values concentrate it sooner.
   * - ``r_max``
     - ``1.0``
     - Maximum exposure rate, in :math:`(0, 1]`. Items at or above it are left
       out before sampling, unless every remaining item is.
```

## Behavior and Comparison

At the last item of a 20-item test at $\theta = 0$, over 20 simulated 500-item
3PL banks, the most informative item is drawn with probability 0.32 to 0.96
(median 0.64).

## Guarantees

1. It never returns an item that was already administered or excluded.
2. When `test_size` is above 1, every available item is equally likely at the first
   position, whatever the acceleration.
3. When every available item gives zero information at the estimate, it draws
   uniformly among them.
4. When every remaining item is at or above the exposure cap, it samples from all
   the remaining items.
5. It draws with the session's random generator, so a run with the same seed
   administers the same items.

## References

```{bibliography}
:filter: docname in docnames
```

:::{seealso}
Companion spec: {doc}`/api/techniques/selection/progressive-selector`
:::

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.ProportionalSelector
   :members:
   :show-inheritance:
```
