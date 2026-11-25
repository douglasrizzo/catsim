# Kullback-Leibler Selector

:Class: {py:class}`catsim.selection.KLSelector`
:Sources: {cite:t}`Chang1996`

## Motivation

Maximum Fisher information is a local criterion: it trusts the current ability
estimate completely. Early in a CAT, that estimate is noisy, so selecting items
only by information at a single point can be brittle.

The KL selector takes a global view. It integrates a
Kullback-Leibler divergence over a shrinking window around the current estimate,
so the method behaves more globally early in the test and more locally later on.

## Definition

For candidate item $i$, current estimate $\hat{\theta}$, and a
comparison point $\theta$, the item-level divergence
{cite:p}`Chang1996{p. 217, Definition 2.1}` is

$$
K_i(\theta \,\|\, \hat{\theta})
=
P_i(\hat{\theta}) \log \frac{P_i(\hat{\theta})}{P_i(\theta)}
+
[1-P_i(\hat{\theta})]
\log \frac{1-P_i(\hat{\theta})}{1-P_i(\theta)}.
$$

It is nonnegative, zero when the two abilities coincide, and its curvature
there is the Fisher information (p. 219, Equation 16). The global criterion
integrates this divergence over a symmetric window (pp. 219–220, Equation 17):

$$
K_i(\hat{\theta})
=
\int_{\hat{\theta}-\delta_n}^{\hat{\theta}+\delta_n}
K_i(\theta \,\|\, \hat{\theta})\,d\theta,
\qquad
\delta_n = \frac{c}{\sqrt{n}},
$$

where $n$ is the number of administered items and $c$, the `window`
parameter, is a positive schedule constant. For a small window the criterion is governed by the
curvature, that is, by the Fisher information; for a large one, by the tails of
the divergence (p. 220). Before the first item, catsim uses $n = 1$, so the
window is $\hat{\theta} \pm c$. Chang and Ying also define a Bayesian version
that weights the divergence by the posterior (p. 222), a planned extension,
KLP. The selector chooses the item, neither administered nor excluded, with the
largest value of $K_i(\hat{\theta})$.

### Computation

Each integral is computed with a Gauss-Legendre rule over the window, the same
rule for every candidate. A steep item's divergence changes quickly near its
difficulty, so the rule's number of nodes is four times the largest
discrimination among the candidates times the half-width $\delta_n$, at least
24 and at most 240. Banks drawn by `ItemBank.generate` with the default window
need 24 to 28 nodes. Up to $a\,\delta_n = 60$, for example $a = 10$ over
$\hat{\theta} \pm 6$, every integral above $10^{-5}$ is within a relative
$10^{-10}$ of adaptive quadrature: at most $7 \times 10^{-11}$ over 2PL and
4PL items with $c \leq 0.3$ and $d \geq 0.85$. Beyond that the rule keeps 240
nodes and the error grows, to about $2 \times 10^{-7}$ at $a\,\delta_n = 120$.

## Parameters

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Parameter
     - Default
     - Description
   * - ``window``
     - ``3.0``
     - Positive constant :math:`c` controlling the half-width :math:`\delta_n = c/\sqrt{n}`.
       :cite:t:`Chang1996` chose :math:`c` from the coverage probability of a
       confidence interval for :math:`\theta` (p. 221) and used :math:`c = 3` in
       their simulation (p. 223).
   * - ``r_max``
     - ``1.0``
     - Maximum exposure rate, in :math:`(0, 1]`. Items at or above it are left
       out before the KL scores are computed, unless every remaining item is.
```

## Guarantees

1. It never returns an item that was already administered or excluded.
2. When every remaining item is at or above the exposure cap, it falls back to the
   remaining item with the largest KL integral.
3. When no item remains, it raises `NoItemsAvailableError`.

## References

```{bibliography}
:filter: docname in docnames
```

:::{seealso}
Posterior-weighted variant: a planned extension, KLP.
:::

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.KLSelector
   :members:
   :show-inheritance:
```
