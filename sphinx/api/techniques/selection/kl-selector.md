**Status:** Implemented
**Module:** {py:class}`catsim.selection.KLSelector`
**Reference:** Chang & Ying (1996)

# Kullback-Leibler Selector

## Motivation

Maximum Fisher information is a local criterion: it trusts the current ability
estimate completely. Early in a CAT, that estimate is noisy, so selecting items
only by information at a single point can be brittle.

The KL selector replaces that local view with a global one. It integrates a
Kullback-Leibler divergence over a shrinking window around the current estimate,
so the method behaves more globally early in the test and more locally later on.

## Definition

For candidate item $i$, current estimate $\hat{\theta}$, and a
comparison point $\theta$, the item-level divergence (Chang & Ying, 1996,
p. 217, Definition 2.1) is

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
KLP. The selector chooses the non-administered item with the
largest value of $K_i(\hat{\theta})$.

Each integral is computed with a 24-point Gauss-Legendre rule over the
window. On banks drawn by `ItemBank.generate`, its
relative error against adaptive quadrature is at most about $10^{-11}$. The
error grows with the discrimination and the width of the window: over
$\hat{\theta} \pm 3$, it is about $5 \times 10^{-9}$ for an item with $a = 4$
and $10^{-6}$ for one with $a = 6$.

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
       Chang and Ying (1996) chose :math:`c` from the coverage probability of a
       confidence interval for :math:`\theta` (p. 221) and used :math:`c = 3` in
       their simulation (p. 223).
   * - ``r_max``
     - ``1.0``
     - Maximum exposure rate, in :math:`(0, 1]`. Items at or above it are left
       out before the KL scores are computed, unless every remaining item is.
```

## Behavioral Contracts

Any correct implementation must satisfy the following properties:

1. The KL integrand evaluates to zero when $\theta = \hat{\theta}$, because a distribution has zero divergence from itself.
2. Increasing the integration half-width increases the accumulated KL mass for the same item and current estimate.
3. The half-width schedule follows $c/\sqrt{\max(1, n)}$ and therefore shrinks as the test length grows.
4. When no items satisfy the exposure cap, the selector falls back to the best non-administered item rather than failing spuriously.
5. As the integration window becomes very narrow, the selector approaches a local-information criterion.

## References

Chang, H.-H., & Ying, Z. (1996). A global information approach to computerized adaptive testing. *Applied Psychological Measurement*, 20(3), 213-229. <https://doi.org/10.1177/014662169602000303>. Open copy: [University of Minnesota Digital Conservancy](https://hdl.handle.net/11299/119092).

:::{seealso}
Posterior-weighted variant: a planned extension, KLP.
:::

## API Reference

```{eval-rst}
.. autoclass:: catsim.selection.KLSelector
   :members:
   :show-inheritance:
```
