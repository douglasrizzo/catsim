"""Concrete selector implementations."""

import numpy
import numpy.typing as npt
from scipy.integrate import quad

from .. import irt
from ..exceptions import NoItemsAvailableError
from ..item_bank import ItemBank
from .base import BaseSelector


class MaxInfoSelector(BaseSelector):
  """Selector that returns the first non-administered item with maximum information for the current theta estimate.

  This is one of the most common item selection methods in CAT, choosing the item
  that provides the most information at the examinee's current estimated ability level.

  Parameters
  ----------
  r_max : float, optional
      Maximum exposure rate for items. Items with exposure rates >= r_max will not
      be selected unless no other items are available. Default is 1 (no restriction).
  """

  def __init__(self, r_max: float = 1) -> None:
    """Initialize a MaxInfoSelector object.

    Parameters
    ----------
    r_max : float, optional
        Maximum exposure rate for items. Default is 1.

    Raises
    ------
    ValueError
        If r_max is not between 0 and 1.
    """
    if not 0 <= r_max <= 1:
      msg = f"r_max must be between 0 and 1, got {r_max}"
      raise ValueError(msg)

    super().__init__()
    self._r_max = float(r_max)

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "Maximum Information Selector"

  @property
  def r_max(self) -> float:
    """Return the maximum exposure rate for items the selector accepts.

    Returns
    -------
    float
        Maximum exposure rate for items the selector accepts.
    """
    return self._r_max

  def select(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,
    rng: numpy.random.Generator | None = None,  # noqa: ARG002
    exposure_rates: npt.NDArray[numpy.floating] | None = None,
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

    Returns
    -------
    int or None
        Index of the next item to be applied or `None` if there are no more items in the item bank.
    """
    # sort items by their information value
    ordered_items = self._sort_by_info(item_bank, est_theta)
    # remove administered ones
    valid_indexes = self._get_non_administered(ordered_items, administered_items)

    if len(valid_indexes) == 0:
      msg = "There are no more items to apply."
      raise NoItemsAvailableError(msg)

    if exposure_rates is None:
      exposure_rates = numpy.zeros(item_bank.n_items, dtype=float)
    valid_indexes_low_r = [idx for idx in valid_indexes if exposure_rates[idx] < self._r_max]
    # return the item with maximum information from the ones available
    return valid_indexes_low_r[0] if len(valid_indexes_low_r) > 0 else valid_indexes[0]


class UrrySelector(BaseSelector):
  """Selector that returns the item whose difficulty parameter is closest to the examinee's ability.

  This method, known as Urry's method, selects items based on the proximity of their
  difficulty (b parameter) to the current ability estimate, which is particularly
  effective for 1PL and 2PL models where information is maximized when b = theta.
  """

  def __init__(self) -> None:
    """Initialize a UrrySelector object."""
    super().__init__()

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "Urry Selector"

  def select(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,
    rng: numpy.random.Generator | None = None,  # noqa: ARG002
    exposure_rates: npt.NDArray[numpy.floating] | None = None,  # noqa: ARG002
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

    Returns
    -------
    int or None
        Index of the next item to be applied or `None` if there are no more items in the item bank.
    """
    ordered_items = self._sort_by_b(item_bank, est_theta)
    valid_indexes = self._get_non_administered(ordered_items, administered_items)

    if len(valid_indexes) == 0:
      msg = "There are no more items to apply."
      raise NoItemsAvailableError(msg)

    return valid_indexes[0]


class IntervalInfoSelector(BaseSelector):
  r"""Selects the item that maximizes the integral of the information function at a predetermined ``interval``.

  The interval is defined by a parameter :math:`\\delta` above and below the current :math:`\\hat\\theta`, like so:
  .. math:: argmax_{i \\in I} \\int_{\\hat\\theta - \\delta}^{\\hat\\theta + \\delta}I_i(\\hat\\theta)
  """

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "Interval Information Selector"

  def __init__(self, interval: float | None = None) -> None:
    r"""Initialize an IntervalInfoSelector object.

    Parameters
    ----------
    interval : float or None, optional
        The interval of the integral. If no interval is passed, the integral is computed from
        :math:`[-\\infty, \\infty]`. Default is None.
    """
    super().__init__()
    self._interval = interval if interval is not None else numpy.inf

  @property
  def interval(self) -> float:
    """Get the size of the interval under which the integral of the information function will be computed."""
    return self._interval

  def select(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,
    rng: numpy.random.Generator | None = None,  # noqa: ARG002
    exposure_rates: npt.NDArray[numpy.floating] | None = None,  # noqa: ARG002
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

    Returns
    -------
    int or None
        Index of the next item to be applied or `None` if there are no more items in the item bank.
    """
    # compute the integral of the information function around an examinee's ability
    information_integral = numpy.array([
      quad(
        irt.inf,
        est_theta - self._interval,
        est_theta + self._interval,
        args=(item[0], item[1], item[2], item[3]),
      )[0]
      for item in item_bank.items
    ])
    # sort by that integral in descending order
    ordered_items = (-information_integral).argsort()
    # remove administered items
    organized_items = self._get_non_administered(ordered_items, administered_items)

    if len(organized_items) == 0:
      msg = "There are no more items to apply."
      raise NoItemsAvailableError(msg)

    return organized_items[0]
