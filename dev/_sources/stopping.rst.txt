Stopping Criteria -- :mod:`catsim.stopping`
*******************************************

All implemented classes in this module inherit from a base abstract class
:py:class:`BaseStopper`. :py:class:`Simulator` allows that a custom stopping criterion be
used during the simulation, as long as it also inherits from
:py:class:`BaseStopper`.

.. inheritance-diagram:: catsim.stopping.BaseStopper catsim.stopping.TestLengthStopper catsim.stopping.MinErrorStopper catsim.stopping.ConfidenceIntervalStopper
   :parts: 1
   :top-classes: catsim._base.Simulable

.. automodule:: catsim.stopping
   :members:
   :show-inheritance:
