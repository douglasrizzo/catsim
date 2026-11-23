"""Stop a test once it reaches a maximum length."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .base import BaseStopper

if TYPE_CHECKING:
  from ..item_bank import ItemBank


@dataclass(frozen=True)
class MaxLengthStopper(BaseStopper):
  """Stop once the test reaches a maximum number of administered items.

  Parameters
  ----------
  max_items : int
      Number of administered items at which the test stops. Must be at least 1.

  Raises
  ------
  ValueError
      If `max_items` is less than 1.

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/stopping/max-length-stopper`.
  """

  max_items: int

  def __post_init__(self) -> None:
    """Validate the maximum length."""
    if self.max_items < 1:
      msg = f"max_items must be at least 1, got {self.max_items}"
      raise ValueError(msg)

  def stop(
    self,
    item_bank: ItemBank,  # noqa: ARG002
    administered_items: list[int],
    theta: float,  # noqa: ARG002
  ) -> bool:
    """Check whether the test reached `max_items` administered items.

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
        True if at least `max_items` items were administered, False otherwise.
    """
    return len(administered_items) >= self.max_items
