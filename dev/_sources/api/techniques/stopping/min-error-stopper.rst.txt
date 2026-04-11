Minimum Error Stopper
=====================

:Status: Implemented
:Module: :py:class:`catsim.stopping.MinErrorStopper`
:Reference: Samejima (1977); Thissen (2000)

Motivation
----------

Variable-length CATs are usually justified by precision: once the current test
is informative enough, continuing to administer items yields diminishing returns.
The minimum-error stopper operationalizes that idea by terminating the test when
the standard error of estimation drops below a target threshold.

This is one of the most common adaptive stopping rules because its threshold is
easy to interpret on the theta scale and directly reflects the information
accumulated by the current administered set.

Definition
----------

Let :math:`I(\hat{\theta})` be the information of the administered test at the
current estimate. The standard error of estimation is

.. math::

   SEE(\hat{\theta})
   =
   \sqrt{\frac{1}{I(\hat{\theta})}}.

After the shared ``min_items``/``max_items``/bank-exhaustion checks inherited
from :class:`~catsim.stopping.TestLengthStopper`, the package stops when

.. math::

   SEE(\hat{\theta}) < \tau,

where :math:`\tau` is the configured ``min_error`` threshold.

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``min_error``
     - required
     - Positive standard-error threshold :math:`\tau`.
   * - ``min_items``
     - ``None``
     - Minimum number of items required before precision-based stopping is allowed.
   * - ``max_items``
     - ``None``
     - Maximum number of items allowed before a hard stop.

Behavioral Contracts
--------------------

Any correct implementation must satisfy the following properties:

1. The stopper returns ``True`` exactly when the computed standard error is strictly less than ``min_error``, after shared hard-stop and gating checks.
2. If no items have been administered, the precision criterion alone cannot stop the test.
3. The stopper requires a finite ``theta`` input and raises ``ValueError`` if ``theta`` is missing.
4. Reaching ``max_items`` still forces termination even if the standard error threshold has not been met.
5. Failing to reach ``min_items`` prevents termination even if the standard error threshold has been met.

References
----------

Samejima, F. (1977). A use of the information function in tailored testing. *Applied Psychological Measurement*, 1(2), 233-247. https://doi.org/10.1177/014662167700100209

Thissen, D. (2000). Reliability and measurement precision. In H. Wainer (Ed.), *Computerized Adaptive Testing: A Primer* (2nd ed., pp. 159-184). Mahwah, NJ: Lawrence Erlbaum Associates.

API Reference
-------------

.. autoclass:: catsim.stopping.MinErrorStopper
   :members:
   :show-inheritance:
