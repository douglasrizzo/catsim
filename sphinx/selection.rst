Item Selection Methods -- :mod:`catsim.selection`
*************************************************

All implemented classes in this module inherit from a base abstract class
:py:class:`BaseSelector`. Selectors can be used manually through
:py:class:`catsim.engine.CatEngine` or through
:py:class:`catsim.simulation.SimulationRunner`, as long as they also inherit from
:py:class:`BaseSelector`.

.. inheritance-diagram:: catsim.selection.BaseSelector catsim.selection.FiniteSelector catsim.selection.MaxInfoSelector catsim.selection.UrrySelector catsim.selection.IntervalInfoSelector catsim.selection.LinearSelector catsim.selection.RandomSelector catsim.selection.RandomesqueSelector catsim.selection.The54321Selector catsim.selection.ClusterSelector catsim.selection.StratifiedSelector catsim.selection.AStratSelector catsim.selection.AStratBBlockSelector catsim.selection.MaxInfoStratSelector catsim.selection.MaxInfoBBlockSelector
   :parts: 1
   :top-classes: catsim.selection.BaseSelector

.. automodule:: catsim.selection
   :members:
   :show-inheritance:
