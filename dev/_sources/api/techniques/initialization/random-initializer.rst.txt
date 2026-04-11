Random Initializer
==================

:Status: Implemented
:Module: :py:class:`catsim.initialization.RandomInitializer`

Motivation
----------

Random initialization samples the initial ability estimate from a configured
distribution. It is useful when simulations should vary the starting point
across examinees while remaining reproducible through the engine or runner
random number generator.

Definition
----------

For the uniform distribution, the initializer samples

.. math::

   \hat{\theta}_0 \sim U(a, b),

where ``dist_params`` provides the two bounds.

For the normal distribution, the initializer samples

.. math::

   \hat{\theta}_0 \sim \mathcal{N}(\mu, \sigma),

where ``dist_params`` provides the mean and standard deviation.

The item bank is accepted to satisfy the initializer contract, but it does not
affect the returned value.

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``dist_type``
     - ``InitializationDistribution.UNIFORM``
     - Distribution used for sampling the initial ability estimate.
   * - ``dist_params``
     - ``(-5, 5)``
     - Distribution parameters. For the uniform distribution, the two values are
       the lower and upper bounds, in either order. For the normal distribution,
       the values are mean and standard deviation, in that order.

Behavioral Contracts
--------------------

Any correct implementation must satisfy the following properties:

1. It validates that ``dist_type`` is an ``InitializationDistribution`` value.
2. It validates that ``dist_params`` contains exactly two values.
3. For uniform initialization, it rejects equal bounds and samples between the lower and upper values.
4. For normal initialization, it rejects non-positive standard deviations.
5. It samples using the random number generator passed to ``initialize``.
6. It does not depend on the item bank.

API Reference
-------------

.. autoclass:: catsim.initialization.RandomInitializer
   :members:
   :show-inheritance:

.. autoclass:: catsim.initialization.InitializationDistribution
   :members:
