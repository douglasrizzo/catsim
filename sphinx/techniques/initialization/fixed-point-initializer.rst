Fixed Point Initializer
=======================

:Status: Implemented
:Module: :py:class:`catsim.initialization.FixedPointInitializer`

Motivation
----------

Fixed-point initialization gives every examinee the same initial ability
estimate. It is useful for deterministic baselines, controlled simulation
studies, and workflows that should not introduce randomness before the first
item is selected.

Definition
----------

For a configured starting value :math:`\theta_0`, the initializer returns the
same value for every examinee:

.. math::

   \hat{\theta}_0 = \theta_0.

The item bank and random number generator are accepted to satisfy the initializer
contract, but they do not affect the returned value.

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``start``
     - Required
     - Ability estimate returned for every examinee.

Behavioral Contracts
--------------------

Any correct implementation must satisfy the following properties:

1. It returns the configured ``start`` value for every initialization request.
2. It does not depend on the item bank.
3. It does not consume random numbers from the provided generator.

API Reference
-------------

.. autoclass:: catsim.initialization.FixedPointInitializer
   :members:
   :show-inheritance:
