"""Base class for CAT stoppers."""

from abc import ABC, abstractmethod

from ..item_bank import ItemBank


class BaseStopper(ABC):
  """Base class for CAT stopping criteria.

  A stopper checks one criterion, such as the test length or the precision of the ability
  estimate, and :class:`~catsim.stopping.StopRule` combines several of them.

  Stoppers are pure: the same inputs always give the same answer, and a stopper keeps no
  per-session state. One engine, and so one stopper, serves every session of a run, and an
  application may interleave the sessions of several examinees on it.
  """

  @abstractmethod
  def stop(self, item_bank: ItemBank, administered_items: list[int], theta: float) -> bool:
    """Check whether the test reached its stopping criterion for the given user.

    Parameters
    ----------
    item_bank : ItemBank
        Item bank used by the session.
    administered_items : list[int]
        Item indices already administered in the session.
    theta : float
        Current ability estimate.

    Returns
    -------
    bool
        True if the test met its stopping criterion, False otherwise.
    """
