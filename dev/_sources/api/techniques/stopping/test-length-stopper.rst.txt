Test Length Stopper
===================

:Status: Implemented
:Module: :py:class:`catsim.stopping.TestLengthStopper`
:Reference: Wainer (2000); Thissen (2000)

Motivation
----------

Fixed-length stopping is the simplest CAT termination rule: the test ends after a
predefined number of administered items. Even when other stopping rules are used,
hard minimum and maximum item limits are often operationally necessary to control
testing burden and guarantee termination.

In :mod:`catsim`, ``TestLengthStopper`` is both the concrete fixed-length rule
and the common base for the more specialized precision-based stoppers.

Definition
----------

Let :math:`m` be the number of administered items. The package checks stopping in
the following order:

.. math::

   \text{stop if } m \ge m_{\max},

.. math::

   \text{stop if } m \ge J,

where :math:`J` is the size of the item bank. If a minimum length
:math:`m_{\min}` is configured and :math:`m < m_{\min}`, the rule returns
``False`` regardless of any subclass-specific criterion. The base
``TestLengthStopper`` has no additional criterion, so after these checks it
returns ``False`` unless one of the hard stops was triggered.

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``min_items``
     - ``None``
     - Minimum number of administered items required before the test may stop.
   * - ``max_items``
     - ``None``
     - Maximum number of administered items allowed before the test must stop.

Behavioral Contracts
--------------------

Any correct implementation must satisfy the following properties:

1. Reaching ``max_items`` is a hard stop regardless of any other criterion.
2. Exhausting the item bank is a hard stop regardless of any other criterion.
3. If ``min_items`` is configured and has not yet been reached, the stopper returns ``False``.
4. The base class requires ``item_bank`` and ``administered_items`` inputs and raises ``ValueError`` if they are missing.
5. The base class itself imposes no precision criterion beyond the shared min/max and exhaustion checks.

References
----------

Wainer, H. (Ed.). (2000). *Computerized Adaptive Testing: A Primer* (2nd ed.). Mahwah, NJ: Lawrence Erlbaum Associates.

Thissen, D. (2000). Reliability and measurement precision. In H. Wainer (Ed.), *Computerized Adaptive Testing: A Primer* (2nd ed., pp. 159-184). Mahwah, NJ: Lawrence Erlbaum Associates.

API Reference
-------------

.. autoclass:: catsim.stopping.TestLengthStopper
   :members:
   :show-inheritance:
