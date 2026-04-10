Kullback-Leibler Selector
=========================

:Status: Planned
:Module: ``catsim.selection.kl.KLSelector`` (planned)
:Reference: Chang & Ying (1996)

Motivation
----------

Maximum Fisher information is a local criterion: it trusts the current ability
estimate completely. Early in a CAT, that estimate is noisy, so selecting items
only by information at a single point can be brittle.

The KL selector replaces that local view with a global one. It integrates a
Kullback-Leibler divergence over a shrinking window around the current estimate,
so the method behaves more globally early in the test and more locally later on.

Definition
----------

For candidate item :math:`i`, current estimate :math:`\hat{\theta}`, and a
comparison point :math:`\theta`, the item-level divergence is

.. math::

   K_i(\theta \,\|\, \hat{\theta})
   =
   P_i(\hat{\theta}) \log \frac{P_i(\hat{\theta})}{P_i(\theta)}
   +
   [1-P_i(\hat{\theta})]
   \log \frac{1-P_i(\hat{\theta})}{1-P_i(\theta)}.

The global criterion integrates this divergence over a symmetric window:

.. math::

   K_i(\hat{\theta})
   =
   \int_{\hat{\theta}-\delta_n}^{\hat{\theta}+\delta_n}
   K_i(\theta \,\|\, \hat{\theta})\,d\theta,
   \qquad
   \delta_n = \frac{c}{\sqrt{n}},

where :math:`n` is the number of administered items and :math:`c` is a positive
schedule constant. The selector chooses the non-administered item with the
largest value of :math:`K_i(\hat{\theta})`.

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Parameter
     - Default
     - Description
   * - ``c``
     - ``3.0``
     - Positive constant controlling the half-width :math:`\delta_n = c/\sqrt{n}`.
       The feature card follows Chang and Ying's recommended range of roughly 3 to 5.
   * - ``r_max``
     - ``1.0``
     - Exposure cap applied after ordering candidate items by KL score.

Behavioral Contracts
--------------------

Any correct implementation must satisfy the following properties:

1. The KL integrand evaluates to zero when :math:`\theta = \hat{\theta}`, because a distribution has zero divergence from itself.
2. Increasing the integration half-width increases the accumulated KL mass for the same item and current estimate.
3. The half-width schedule follows :math:`c/\sqrt{\max(1, n)}` and therefore shrinks as the test length grows.
4. When no items satisfy the exposure cap, the selector falls back to the best non-administered item rather than failing spuriously.
5. As the integration window becomes very narrow, the selector approaches a local-information criterion.

References
----------

Chang, H.-H., & Ying, Z. (1996). A global information approach to computerized adaptive testing. *Applied Psychological Measurement*, 20(3), 213-229. https://doi.org/10.1177/014662169602000303

.. seealso::

   Posterior-weighted variant planned in card 15: KLP selector.
