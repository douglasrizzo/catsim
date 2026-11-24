"""Stop a test once it reaches a maximum length."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from typing_extensions import override

from .base import BaseStopper

if TYPE_CHECKING:
  from ..context import StepContext


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

  @override
  def stop(self, context: StepContext) -> bool:
    """Return whether at least `max_items` items have been administered.

    Parameters
    ----------
    context : StepContext
        The session to check.

    Returns
    -------
    bool
        True if at least `max_items` items were administered, False otherwise.
    """
    return len(context.administered_items) >= self.max_items
