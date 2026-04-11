Progressive Selector
====================

:Status: Planned
:Module: ``catsim.selection.progressive.ProgressiveSelector`` (planned)
:Reference: Revuelta & Ponsoda (1998); Barrada et al. (2008, 2010)

Motivation
----------

Pure maximum-information selection tends to overuse a small set of highly
informative items. Progressive selection addresses that exposure imbalance by
starting almost random and gradually moving toward information-driven scoring as
the test advances.

This gives the method a clear operational role: better exposure control early,
near-Fisher behavior late, and no need for offline calibration.

Definition
----------

At test position :math:`n` out of a fixed total length :math:`N`, the candidate
score for item :math:`i` is

.. math::

   S_i^{(n)}
   =
   (1-w_n)\,R_i
   +
   w_n\,\frac{I_i(\hat{\theta})}{\max_j I_j(\hat{\theta})},

where :math:`R_i` is a fresh uniform random draw on :math:`[0,1]` and

.. math::

   w_n
   =
   \left(\frac{n-1}{N-1}\right)^s,

with acceleration parameter :math:`s > 0`. The selector chooses the
non-administered item with the largest score.

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Parameter
     - Default
     - Description
   * - ``test_size``
     - required
     - Planned total number of administered items :math:`N`; required because the schedule depends on test position.
   * - ``acceleration``
     - ``1.0``
     - Positive exponent :math:`s`. Values above 1 delay the transition to information-based selection; values below 1 accelerate it.
   * - ``r_max``
     - ``1.0``
     - Exposure cap applied after scoring candidates.

Behavioral Contracts
--------------------

Any correct implementation must satisfy the following properties:

1. The blend weight is exactly 0 at position 1 and exactly 1 at the final test position when ``test_size > 1``.
2. With ``acceleration = 1``, the schedule is linear in test position; larger acceleration delays the shift toward information-based scoring and smaller acceleration speeds it up.
3. At the first position, selection is driven only by the random component and is therefore independent of item information.
4. At the final position, the score reduces to normalized information and behaves like an information-based selector subject to the exposure rule.
5. The random component must be drawn from the provided RNG when one is supplied so repeated simulations remain reproducible.

References
----------

Revuelta, J., & Ponsoda, V. (1998). A comparison of item exposure control methods in computerized adaptive testing. *Journal of Educational Measurement*, 35(4), 311-327.

Barrada, J. R., Olea, J., Ponsoda, V., & Abad, F. J. (2008). Incorporating randomness in the Fisher information function for item exposure control. *Methodology*, 4(2), 51-59.

Barrada, J. R., Olea, J., Ponsoda, V., & Abad, F. J. (2010). A method for the comparison of item selection rules in computerized adaptive testing. *Applied Psychological Measurement*, 34(6), 438-452.

.. seealso::

   Companion planned spec: :doc:`/api/techniques/selection/proportional-selector`
