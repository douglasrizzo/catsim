Confidence Interval Stopper
===========================

:Status: Implemented
:Module: :py:class:`catsim.stopping.ConfidenceIntervalStopper`
:Reference: Thissen (2000); Wainer (2000)

Motivation
----------

Some adaptive tests are used for categorical decisions rather than for reporting a
single continuous ability estimate. In that setting it is often more natural to
stop once the uncertainty around :math:`\hat{\theta}` falls entirely within one
predefined ability band, because the classification is then stable.

The confidence-interval stopper in :mod:`catsim` implements exactly that
classification-oriented precision rule on top of the shared length and exhaustion
constraints.

Definition
----------

The package computes a normal-approximation confidence interval

.. math::

   CI(\hat{\theta})
   =
   \left[
     \hat{\theta} - z_{(1+\gamma)/2} \, SEE(\hat{\theta}),
     \hat{\theta} + z_{(1+\gamma)/2} \, SEE(\hat{\theta})
   \right],

where :math:`\gamma` is the configured confidence level and
:math:`SEE(\hat{\theta}) = \sqrt{1 / I(\hat{\theta})}`.

Given sorted boundary points :math:`b_1 < \dots < b_m`, the rule stops when the
entire confidence interval falls within one interval induced by those boundaries:

.. math::

   (-\infty, b_1],\ [b_1, b_2],\ \dots,\ [b_m, \infty).

Operationally, the implementation checks whether the upper bound lies below the
first boundary, the lower bound lies above the last boundary, or both endpoints
lie between the same adjacent pair of boundaries.

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``interval_bounds``
     - required
     - Sorted boundary points that define the ability bands used for classification.
   * - ``confidence``
     - ``0.95``
     - Confidence level :math:`\gamma` used to form the normal-approximation interval.
   * - ``min_items``
     - ``None``
     - Minimum number of items required before the classification rule may stop the test.
   * - ``max_items``
     - ``None``
     - Maximum number of items allowed before a hard stop.

Behavioral Contracts
--------------------

Any correct implementation must satisfy the following properties:

1. ``interval_bounds`` must be nonempty and sorted in ascending order.
2. The stopper returns ``True`` only when the full confidence interval lies within a single induced classification interval, after shared hard-stop and gating checks.
3. If the confidence interval crosses any configured boundary, the stopper returns ``False``.
4. The stopper requires ``theta`` and raises ``ValueError`` if it is missing.
5. Reaching ``max_items`` or exhausting the item bank still forces termination even if the confidence-interval rule is not met.

References
----------

Thissen, D. (2000). Reliability and measurement precision. In H. Wainer (Ed.), *Computerized Adaptive Testing: A Primer* (2nd ed., pp. 159-184). Mahwah, NJ: Lawrence Erlbaum Associates.

Wainer, H. (Ed.). (2000). *Computerized Adaptive Testing: A Primer* (2nd ed.). Mahwah, NJ: Lawrence Erlbaum Associates.

API Reference
-------------

.. autoclass:: catsim.stopping.ConfidenceIntervalStopper
   :members:
   :show-inheritance:
