"""Base class for CAT stoppers."""

from abc import ABC, abstractmethod

from ..context import StepContext


class BaseStopper(ABC):
  """Base class for CAT stopping criteria.

  A stopper checks one criterion, such as the test length or the precision of the ability
  estimate, against what a :class:`~catsim.StepContext` holds, and
  :class:`~catsim.stopping.StopRule` combines several of them.

  Stoppers are pure: the same context always gives the same answer. A stopper holds its
  configuration and caches derived from the item bank, never per-session state: one stopper serves
  every session of a run, which an application may interleave.
  """

  @abstractmethod
  def stop(self, context: StepContext) -> bool:
    """Return whether the session has met the stopping criterion.

    Parameters
    ----------
    context : StepContext
        The session to check, with ``theta`` its current estimate. When
        :meth:`CatEngine.answer <catsim.CatEngine.answer>` calls the stopper, the context's
        generator is created for the call with a fixed seed.

    Returns
    -------
    bool
        True if the test met its stopping criterion, False otherwise.
    """
