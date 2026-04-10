Initialization Methods -- :mod:`catsim.initialization`
******************************************************

All implemented classes in this module inherit from a base abstract class
:py:class:`BaseInitializer`. Initializers can be used manually through
:py:class:`catsim.engine.CatEngine` or through
:py:class:`catsim.simulation.SimulationRunner`, as long as they also inherit from
:py:class:`BaseInitializer`.

.. inheritance-diagram:: catsim.initialization.BaseInitializer catsim.initialization.RandomInitializer catsim.initialization.FixedPointInitializer
   :parts: 1
   :top-classes: catsim.initialization.BaseInitializer

.. automodule:: catsim.initialization
    :members:
    :show-inheritance:
