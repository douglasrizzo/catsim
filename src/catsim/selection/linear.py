"""Linear selector implementation."""

import numpy
import numpy.typing as npt

from ..exceptions import NoItemsAvailableError
from ..item_bank import ItemBank
from .base import FiniteSelector


class LinearSelector(FiniteSelector):
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

    super().__init__(len(indexes))
    self._indexes = indexes
    self._current = 0

  @property
  def indexes(self) -> list[int]:
    """The indexes of the items that will be returned in order."""
    return self._indexes

  @property
  def current(self) -> int:
    """The index of the current item."""
    return self._current

  def select(
    self,
    item_bank: ItemBank,  # noqa: ARG002
    administered_items: list[int],
    est_theta: float,  # noqa: ARG002
    rng: numpy.random.Generator | None = None,  # noqa: ARG002
    exposure_rates: npt.NDArray[numpy.floating] | None = None,  # noqa: ARG002
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
    int or None
        Index of the next item to be applied or `None` if there are no more items in the item bank.
    """
    valid_indexes = self._get_non_administered(self._indexes, administered_items)
    if len(valid_indexes) == 0:
      msg = (
        f"A new index was asked for, but there are no more item indexes to present.\n"
        f"Current item:\t\t\t{self._current}\n"
        f"Items to be administered:\t{sorted(self._indexes)} (size: {len(self._indexes)})\n"
        f"Administered items:\t\t{sorted(administered_items)} (size: {len(administered_items)})"
      )
      raise NoItemsAvailableError(msg)
    return valid_indexes[0]
