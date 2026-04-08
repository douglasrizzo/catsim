"""Base class for CAT stoppers."""

from abc import ABC, abstractmethod
from typing import Any

from ..item_bank import ItemBank


class BaseStopper(ABC):
  """Base class for CAT stopping criteria.

  Stoppers determine when a test should end based on specific criteria such as
  test length, measurement precision, or other conditions.
  """

  @abstractmethod
  def stop(self, item_bank: ItemBank, administered_items: list[int], theta: float | None, **kwargs: Any) -> bool:
    """Check whether the test reached its stopping criterion for the given user.

    Parameters
    ----------
    item_bank : ItemBank
        Item bank used by the session.
    administered_items : list[int]
        Item indices already administered in the session.
    theta : float or None
        Current ability estimate.
    **kwargs : dict
        Additional keyword arguments that specific Stopper implementations may require.

    Returns
    -------
    bool
        True if the test met its stopping criterion, False otherwise.
    """
