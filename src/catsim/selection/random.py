"""Random selector implementations."""

import numpy.typing as npt
from numpy.random import Generator

from ..exceptions import NoItemsAvailableError
from ..item_bank import ItemBank
from .base import BaseSelector


class RandomSelector(BaseSelector):
  """Selector that randomly selects items for application.

  This selector is useful for baseline comparisons or for studying the impact of
  item selection strategies.

  Parameters
  ----------
  replace : bool, optional
      Whether to select an item that has already been selected before for this examinee.
      Default is False.
  """

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "Random Selector"

  def __init__(self, replace: bool = False) -> None:
    """Initialize a RandomSelector object.

    Parameters
    ----------
    replace : bool, optional
        Whether to select an item that has already been selected before for this examinee.
        Default is False.
    """
    super().__init__()
    self._replace = replace

  def select(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,  # noqa: ARG002
    rng: Generator | None = None,
    exposure_rates: npt.NDArray | None = None,  # noqa: ARG002
  ) -> int | None:
    """Return the index of the next item to be administered.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    administered_items : list[int]
        A list containing the indexes of items that were already administered.
    est_theta : float
        Current estimated ability. Unused by this selector.
    rng : numpy.random.Generator or None, optional
        Random number generator used by the object.

    Returns
    -------
    int or None
        Index of the next item to be applied or `None` if there are no more items in the item bank.
    """
    if rng is None:
      msg = "rng parameter cannot be None"
      raise ValueError(msg)

    if len(administered_items) >= item_bank.n_items and not self._replace:
      msg = "There are no more items to apply."
      raise NoItemsAvailableError(msg)

    if self._replace:
      return int(rng.choice(item_bank.n_items))
    valid_indexes = self._get_non_administered(list(range(item_bank.n_items)), administered_items)
    return int(rng.choice(valid_indexes))


class The54321Selector(BaseSelector):
  r"""Implementation of the 5-4-3-2-1 selector proposed by :cite:p:`McBride83`.

  The first item is drawn at random from the 5 most informative items at the current :math:`\hat\theta`,
  the second from the 4 most informative remaining items, and so on, so that from the fifth item on the
  most informative item is administered. Randomizing the opening items, which every examinee would
  otherwise share while :math:`\hat\theta` is still rough, reduces their exposure at little cost in
  precision.

  Parameters
  ----------
  initial_bin_size : int, optional
      Number of most informative items the first item is drawn from; each following item draws from
      one fewer. Default is 5.

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/selection/the-54321-selector`.
  """

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "5-4-3-2-1 Selector"

  def __init__(self, initial_bin_size: int = 5) -> None:
    """Initialize a The54321Selector object.

    Parameters
    ----------
    initial_bin_size : int, optional
        Number of most informative items the first item is drawn from. Default is 5.

    Raises
    ------
    ValueError
        If ``initial_bin_size`` is smaller than 1.
    """
    if initial_bin_size < 1:
      msg = f"initial_bin_size must be at least 1, got {initial_bin_size}"
      raise ValueError(msg)
    super().__init__()
    self._initial_bin_size = initial_bin_size

  @property
  def initial_bin_size(self) -> int:
    """Get the number of most informative items the first item is drawn from."""
    return self._initial_bin_size

  def select(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,
    rng: Generator | None = None,
    exposure_rates: npt.NDArray | None = None,  # noqa: ARG002
  ) -> int | None:
    """Return the index of the next item to be administered.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    administered_items : list[int]
        A list containing the indexes of items that were already administered.
    est_theta : float
        The current estimated ability.
    rng : numpy.random.Generator or None, optional
        Random number generator used by the object.

    Returns
    -------
    int or None
        Index of the next item to be applied or `None` if there are no more items in the item bank.
    """
    if rng is None:
      msg = "rng parameter cannot be None"
      raise ValueError(msg)

    # sort item indexes by their information value descending and remove indexes of administered items
    ordered_items = self._sort_by_info(item_bank, est_theta)
    organized_items = self._get_non_administered(ordered_items, administered_items)

    if len(organized_items) == 0:
      msg = "There are no more items to apply."
      raise NoItemsAvailableError(msg)

    bin_size = max(1, self._initial_bin_size - len(administered_items))
    return int(rng.choice(organized_items[:bin_size]))


class RandomesqueSelector(BaseSelector):
  """Implementation of the randomesque selector proposed by :cite:p:`Kingsbury89`.

  In this selector, at each step of the test, an item is randomly chosen from the :math:`n` most informative items in
  the item bank, :math:`n` being a user-defined value.
  """

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "Randomesque Selector"

  def __init__(self, bin_size: int) -> None:
    """Initialize a RandomesqueSelector object.

    Parameters
    ----------
    bin_size : int
        The number of most informative items to be taken into consideration when
        randomly selecting one of them.
    """
    super().__init__()
    self._bin_size = bin_size

  @property
  def bin_size(self) -> int:
    """Get the bin size."""
    return self._bin_size

  def select(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,
    rng: Generator | None = None,
    exposure_rates: npt.NDArray | None = None,  # noqa: ARG002
  ) -> int | None:
    """Return the index of the next item to be administered.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    administered_items : list[int]
        A list containing the indexes of items that were already administered.
    est_theta : float
        The current estimated ability.
    rng : numpy.random.Generator or None, optional
        Random number generator used by the object.

    Returns
    -------
    int or None
        Index of the next item to be applied or `None` if there are no more items in the item bank.
    """
    if rng is None:
      msg = "rng parameter cannot be None"
      raise ValueError(msg)

    # sort item indexes by their information value descending and remove indexes of administered items
    ordered_items = self._sort_by_info(item_bank, est_theta)
    organized_items = self._get_non_administered(ordered_items, administered_items)

    if len(organized_items) == 0:
      msg = "There are no more items to apply."
      raise NoItemsAvailableError(msg)

    return int(rng.choice(list(organized_items)[: self._bin_size]))
