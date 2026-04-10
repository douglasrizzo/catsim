Stopping Criteria -- :mod:`catsim.stopping`
*******************************************

For algorithmic details, mathematical definitions, and behavioral contracts of each
stopping criterion, see :doc:`/techniques/stopping/index`.

All implemented classes in this module inherit from a base abstract class
:py:class:`BaseStopper`. Stoppers can be used manually through
:py:class:`catsim.engine.CatEngine` or through
:py:class:`catsim.simulation.SimulationRunner`, as long as they also inherit from
:py:class:`BaseStopper`.

.. inheritance-diagram:: catsim.stopping.BaseStopper catsim.stopping.TestLengthStopper catsim.stopping.MinErrorStopper catsim.stopping.ConfidenceIntervalStopper
   :parts: 1
   :top-classes: catsim.stopping.BaseStopper

.. automodule:: catsim.stopping
   :members:
   :show-inheritance:
   :no-index:
