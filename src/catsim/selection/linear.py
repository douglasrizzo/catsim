"""Linear selector implementation."""

from typing_extensions import override

from ..context import StepContext
from ..exceptions import NoItemsAvailableError
from .base import BaseSelector, remaining


class LinearSelector(BaseSelector):
  """Selector that administers a fixed list of items in order, simulating a standard (non-adaptive) test.

  This selector is useful for baseline comparisons or for administering a fixed set
  of items in a predetermined order.

  Parameters
  ----------
  items : list[int]
      Indices of the items to administer, in order. They are checked against the item bank at
      every selection.

  Raises
  ------
  ValueError
      If ``items`` is empty or holds a negative index.
  """

  @override
  def __repr__(self) -> str:
    """Return the selector as a constructor call that names every argument."""
    return f"{type(self).__name__}(items={self._items!r})"

  def __init__(self, items: list[int]) -> None:
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

  @override
  def select(self, context: StepContext) -> int:
    """Return the first of the selector's items that is neither administered nor excluded.

    Parameters
    ----------
    context : StepContext
        The session to select for.

    Returns
    -------
    int
        Index of the selected item.

    Raises
    ------
    ValueError
        If any of the selector's items is not an index of ``context.item_bank``.
    NoItemsAvailableError
        If every one of the selector's items has been administered or excluded.
    """
    n_items = context.item_bank.n_items
    outside = [item for item in self._items if item >= n_items]
    if outside:
      msg = f"items {outside} are not in the item bank, whose indices run from 0 to {n_items - 1}"
      raise ValueError(msg)

    available = set(remaining(context).tolist())
    item = next((candidate for candidate in self._items if candidate in available), None)
    if item is None:
      msg = (
        "Every item of the selector's list has been administered or excluded.\n"
        f"Items to be administered:\t{sorted(self._items)} (size: {len(self._items)})\n"
        f"Administered items:\t\t{sorted(context.administered_items)} (size: {len(context.administered_items)})\n"
        f"Excluded items:\t\t{sorted(context.excluded_items)} (size: {len(context.excluded_items)})"
      )
      raise NoItemsAvailableError(msg)
    return int(item)
