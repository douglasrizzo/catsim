54321 Selector
==============

:Status: Implemented
:Module: :py:class:`catsim.selection.The54321Selector`

Specification
-------------

``The54321Selector`` is a finite ordered selector that administers items by a
precomputed difficulty schedule. The schedule steps through five difficulty
levels and selects available items in that deterministic order.

This rule is primarily useful as a reproducible non-adaptive baseline.

Behavioral Contracts
--------------------

Any correct implementation must satisfy the following properties:

1. It never returns an already administered item.
2. It follows the deterministic difficulty-level ordering computed for the item bank.
3. If no non-administered items remain, it raises ``NoItemsAvailableError``.

API Reference
-------------

.. autoclass:: catsim.selection.The54321Selector
   :members:
   :show-inheritance:
