"""Base class for CAT stoppers."""

from abc import ABC, abstractmethod

from ..context import StepContext


class BaseStopper(ABC):
  """Base class for CAT stopping criteria.

  A stopper checks one criterion, such as the test length or the precision of the ability
  estimate, against what a :class:`~catsim.context.StepContext` holds, and
  :class:`~catsim.stopping.StopRule` combines several of them.

  Stoppers are pure: the same context always gives the same answer. A stopper holds its
  configuration and caches derived from the item bank, never per-session state: one engine, and so
  one stopper, serves every session of a run, and an application may interleave the sessions of
  several examinees on it.
  """

  @abstractmethod
  def stop(self, context: StepContext) -> bool:
    """Return whether the session has met the stopping criterion.

    Parameters
    ----------
    context : StepContext
        The session to check, with ``theta`` its current estimate.

    Returns
    -------
    bool
        True if the test met its stopping criterion, False otherwise.
    """
