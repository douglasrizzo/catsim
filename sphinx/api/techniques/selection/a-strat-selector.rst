A-Stratified Selector
=====================

:Status: Implemented
:Module: :py:class:`catsim.selection.AStratSelector`
:Reference: Chang & Ying (1999)

Motivation
----------

The a-stratified method reduces overexposure of highly discriminating items by
reserving them for later stages of the test. Early stages use lower-
discrimination items, when the ability estimate is still unstable; later stages
unlock higher-discrimination items once :math:`\hat{\theta}` is more reliable.

The implementation in :mod:`catsim` follows the same high-level motivation with a
simple stage-based ordering rule built on the generic stratified family base.

Definition
----------

Let :math:`a_i` be the discrimination parameter of item :math:`i`. The package
implementation first presorts all items by increasing discrimination:

.. math::

   a_{(1)} \le a_{(2)} \le \dots \le a_{(J)}.

The presorted bank is then partitioned into ``test_size`` contiguous strata using
the generic stratified slicing rule. At test position :math:`k+1`, the selector
returns the first non-administered item from stratum :math:`k`.

This is a simplified operational form of the a-stratified idea: strata are based
on ascending :math:`a`, and selection inside the active stratum follows the fixed
presorted order.

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``test_size``
     - required
     - Number of strata and intended test length.

Behavioral Contracts
--------------------

Any correct implementation must satisfy the following properties:

1. The bank is presorted once in ascending order of item discrimination.
2. Early stages can only draw from lower-discrimination strata and later stages from higher-discrimination strata.
3. The selector never returns an already administered item.
4. Selection inside a stratum is deterministic given the presorted order and administered-item set.
5. If the active stratum has been exhausted, the selector raises ``NoItemsAvailableError``.

References
----------

Chang, H.-H., & Ying, Z. (1999). A-stratified multistage computerized adaptive testing. *Applied Psychological Measurement*, 23(3), 211-221. https://doi.org/10.1177/01466219922031338

API Reference
-------------

.. autoclass:: catsim.selection.AStratSelector
   :members:
   :show-inheritance:
