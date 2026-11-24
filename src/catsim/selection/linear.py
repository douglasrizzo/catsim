"""Linear selector implementation."""

import numpy

from ..exceptions import NoItemsAvailableError
from ..item_bank import ItemBank
from ..typing import FloatArray
from .base import BaseSelector


class LinearSelector(BaseSelector):
  """Selector that administers a fixed list of items in order, simulating a standard (non-adaptive) test.

  This selector is useful for baseline comparisons or for administering a fixed set
  of items in a predetermined order.

  Parameters
  ----------
  items : list[int]
      Indices of the items to administer, in order. They are checked against the item bank at
      every selection.
  """

  def __repr__(self) -> str:
    """Return the selector as a constructor call that names every argument."""
    return f"{type(self).__name__}(items={self._items!r})"

  def __init__(self, items: list[int]) -> None:
    """Initialize a LinearSelector object.

    Parameters
    ----------
    items : list[int]
        Indices of the items to administer, in order.

    Raises
    ------
    ValueError
        If ``items`` is empty or holds a negative index.
    """
    if len(items) == 0:
      msg = "items cannot be empty"
      raise ValueError(msg)
    if min(items) < 0:
      msg = f"item indices must be non-negative, got {min(items)}"
      raise ValueError(msg)

    super().__init__()
    self._items = list(items)

  @property
  def items(self) -> list[int]:
    """Get the indices of the items to administer, in order."""
    return list(self._items)

  def select(
    self,
    item_bank: ItemBank,
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
        The item bank the selector's items must belong to.
    administered_items : list[int]
        A list containing the indexes of items that were already administered.
    est_theta : float
        Unused: the order of the items is fixed.
    rng : numpy.random.Generator or None, optional
        Unused: the order of the items is fixed.
    exposure_rates : numpy.ndarray or None, optional
        Ignored: this selector applies no exposure cap.
    response_vector : list[bool] or None, optional
        Unused; accepted for the engine's selector contract.

    Returns
    -------
    int
        Index of the next item to be applied: the first of the selector's items not yet administered.

    Raises
    ------
    ValueError
        If any of the selector's items is not an index of ``item_bank``.
    NoItemsAvailableError
        If every one of the selector's items has already been administered.
    """
    outside = [item for item in self._items if item >= item_bank.n_items]
    if outside:
      msg = f"items {outside} are not in the item bank, whose indices run from 0 to {item_bank.n_items - 1}"
      raise ValueError(msg)

    administered = set(administered_items)
    item = next((candidate for candidate in self._items if candidate not in administered), None)
    if item is None:
      msg = (
        "Every item of the selector's list has been administered.\n"
        f"Items to be administered:\t{sorted(self._items)} (size: {len(self._items)})\n"
        f"Administered items:\t\t{sorted(administered_items)} (size: {len(administered_items)})"
      )
      raise NoItemsAvailableError(msg)
    return int(item)
