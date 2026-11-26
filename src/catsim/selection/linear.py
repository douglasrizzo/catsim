"""Linear selector implementation."""

from ..compat import override
from ..context import StepContext
from ..exceptions import NoItemsAvailableError
from .base import BaseSelector, remaining


class LinearSelector(BaseSelector):
  """Selector that administers a fixed list of items in order, simulating a standard (non-adaptive) test.

  This selector is useful for baseline comparisons or for administering a fixed set
  of items in a predetermined order.

  Parameters
  ----------
  indices : list[int]
      Indices of the items to administer, in order. They are checked against the item bank at
      every selection.

  Raises
  ------
  ValueError
      If ``indices`` is empty or holds a negative index.
  """

  @override
  def __repr__(self) -> str:
    """Return the selector as a constructor call that names every argument."""
    return f"{type(self).__name__}(indices={self._indices!r})"

  def __init__(self, indices: list[int]) -> None:
    if len(indices) == 0:
      msg = "indices cannot be empty"
      raise ValueError(msg)
    if min(indices) < 0:
      msg = f"item indices must be non-negative, got {min(indices)}"
      raise ValueError(msg)

    super().__init__()
    self._indices = list(indices)

  @property
  def indices(self) -> list[int]:
    """Get the indices of the items to administer, in order."""
    return list(self._indices)

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
    outside = [item for item in self._indices if item >= n_items]
    if outside:
      msg = f"indices {outside} are not in the item bank, whose indices run from 0 to {n_items - 1}"
      raise ValueError(msg)

    available = set(remaining(context).tolist())
    item = next((candidate for candidate in self._indices if candidate in available), None)
    if item is None:
      msg = (
        "Every item of the selector's list has been administered or excluded.\n"
        f"Items to be administered:\t{sorted(self._indices)} (size: {len(self._indices)})\n"
        f"Administered items:\t\t{sorted(context.administered_items)} (size: {len(context.administered_items)})\n"
        f"Excluded items:\t\t{sorted(context.excluded_items)} (size: {len(context.excluded_items)})"
      )
      raise NoItemsAvailableError(msg)
    return int(item)
