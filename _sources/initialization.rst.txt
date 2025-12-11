Initialization Methods -- :mod:`catsim.initialization`
******************************************************

All implemented classes in this module inherit from a base abstract class
:py:class:`BaseInitializer`. :py:class:`Simulator` allows that a custom initializer be
used during the simulation, as long as it also inherits from
:py:class:`BaseInitializer`.

.. inheritance-diagram:: catsim.initialization.BaseInitializer catsim.initialization.RandomInitializer catsim.initialization.FixedPointInitializer
   :parts: 1
   :top-classes: catsim._base.Simulable

.. automodule:: catsim.initialization
    :members:
    :show-inheritance:
