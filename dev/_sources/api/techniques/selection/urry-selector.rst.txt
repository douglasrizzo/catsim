Urry Selector
=============

:Status: Implemented
:Module: :py:class:`catsim.selection.UrrySelector`
:Reference: Urry (1977); Lord (1980)

Motivation
----------

The Urry selector, also called a ``b``-matching rule, chooses the item whose
difficulty is closest to the current ability estimate. This criterion is easy to
compute, avoids the stronger preference for highly discriminating items shown by
maximum-information rules, and is historically tied to early tailored testing
systems.

For 1PL and 2PL item models, matching difficulty to ability is closely aligned
with maximizing information. In 3PL and 4PL settings the equivalence no longer
holds exactly, but the rule remains a simple and interpretable heuristic.

Definition
----------

Let :math:`b_i` be the difficulty parameter of item :math:`i`. The selector
chooses

.. math::

   i^*
   =
   \arg\min_{i \in R_k} \left| b_i - \hat{\theta} \right|,

where :math:`R_k` is the set of non-administered items. The current package
implementation sorts all items by absolute distance from :math:`\hat{\theta}` to
their :math:`b` values and returns the first remaining item.

Parameters
----------

This selector has no constructor parameters.

Behavioral Contracts
--------------------

Any correct implementation must satisfy the following properties:

1. It ranks items by increasing absolute distance between item difficulty and ``est_theta``.
2. It never returns an already administered item.
3. It ignores exposure-rate inputs; only difficulty proximity controls the choice.
4. In Rasch/1PL settings, the chosen item coincides with a maximum-information choice because information is maximized near :math:`b_i = \hat{\theta}`.
5. If no non-administered items remain, it raises ``NoItemsAvailableError``.

References
----------

Urry, V. W. (1977). Tailored testing: A successful application of latent trait theory. *Journal of Educational Measurement*, 14(2), 181-196. https://doi.org/10.1111/j.1745-3984.1977.tb00035.x

Lord, F. M. (1980). *Applications of Item Response Theory to Practical Testing Problems*. Hillsdale, NJ: Lawrence Erlbaum Associates.

API Reference
-------------

.. autoclass:: catsim.selection.UrrySelector
   :members:
   :show-inheritance:
