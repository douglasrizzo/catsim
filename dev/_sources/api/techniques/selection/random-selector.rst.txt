Random Selector
===============

:Status: Implemented
:Module: :py:class:`catsim.selection.RandomSelector`

Specification
-------------

``RandomSelector`` selects uniformly at random from the non-administered item
pool. It is a simple stochastic baseline for comparing adaptive selection
techniques against a non-adaptive item choice rule.

Behavioral Contracts
--------------------

Any correct implementation must satisfy the following properties:

1. It samples only from non-administered items.
2. It uses the configured random number generator when one is provided.
3. If no non-administered items remain, it raises ``NoItemsAvailableError``.

API Reference
-------------

.. autoclass:: catsim.selection.RandomSelector
   :members:
   :show-inheritance:
