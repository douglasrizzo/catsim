"""Linear selector implementation."""

import numpy

from .._typing import FloatArray
from ..exceptions import NoItemsAvailableError
from ..item_bank import ItemBank
from .base import BaseSelector


class LinearSelector(BaseSelector):
  """Selector that returns item indexes in a linear order, simulating a standard (non-adaptive) test.

  This selector is useful for baseline comparisons or for administering a fixed set
  of items in a predetermined order.

  Parameters
  ----------
  indexes : list[int]
      The indexes of the items that will be returned in order.
  """

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "Linear Selector"

  def __init__(self, indexes: list[int]) -> None:
    """Initialize a LinearSelector object.

    Parameters
    ----------
    indexes : list[int]
        List of item indices to be administered in order.

    Raises
    ------
    ValueError
        If indexes list is empty or contains invalid (negative) values.
    """
    if len(indexes) == 0:
      msg = "indexes list cannot be empty"
      raise ValueError(msg)
    if not all(i >= 0 for i in indexes):
      msg = "All indexes must be non-negative integers"
      raise ValueError(msg)

    super().__init__()
    self._indexes = indexes

  @property
  def indexes(self) -> list[int]:
    """The indexes of the items that will be returned in order."""
    return self._indexes

  def select(
    self,
    item_bank: ItemBank,  # noqa: ARG002
    administered_items: list[int],
    est_theta: float,  # noqa: ARG002
    rng: numpy.random.Generator | None = None,  # noqa: ARG002
    exposure_rates: FloatArray | None = None,  # noqa: ARG002
    response_vector: list[bool] | None = None,  # noqa: ARG002
  ) -> int | None:
    """Return the index of the next item to be administered.

    Parameters
    ----------
    item_bank : ItemBank
        Unused for linear selection.
    administered_items : list[int]
        A list containing the indexes of items that were already administered.

    Returns
    -------
    int
        Index of the next item to be applied: the first of the selector's indexes not yet administered.

    Raises
    ------
    NoItemsAvailableError
        If every one of the selector's indexes has already been administered.
    """
    administered = set(administered_items)
    item = next((index for index in self._indexes if index not in administered), None)
    if item is None:
      msg = (
        f"A new index was asked for, but there are no more item indexes to present.\n"
        f"Items to be administered:\t{sorted(self._indexes)} (size: {len(self._indexes)})\n"
        f"Administered items:\t\t{sorted(administered_items)} (size: {len(administered_items)})"
      )
      raise NoItemsAvailableError(msg)
    return int(item)
