"""Base class for CAT selectors."""

from abc import ABC, abstractmethod

import numpy
import numpy.typing as npt
from numpy.random import Generator

from ..exceptions import NoItemsAvailableError
from ..item_bank import ItemBank


class BaseSelector(ABC):
  """Base class representing a CAT item selector.

  Selectors are responsible for choosing which item to administer next to an
  examinee based on their current estimated ability and test progress.
  """

  @staticmethod
  def _get_non_administered(item_indices: list[int], administered_item_indices: list[int]) -> list[int]:
    """Get a list of items that were not administered from a list of indices.

    Parameters
    ----------
    item_indices : list[int]
        A list of integers corresponding to item indices.
    administered_item_indices : list[int]
        A list of integers corresponding to the indices of items that were already
        administered to a given examinee.

    Returns
    -------
    list[int]
        A list of items corresponding to the indices that are in `item_indices` but
        not in `administered_item_indices`, in the same order they were passed in
        `item_indices`.
    """
    return [x for x in item_indices if x not in administered_item_indices]

  @staticmethod
  def _exposure_filtered_candidates(
    n_items: int,
    administered_items: list[int],
    exposure_rates: npt.NDArray[numpy.floating] | None,
    r_max: float,
  ) -> npt.NDArray[numpy.intp]:
    """Return non-administered item indices, preferring items below the exposure cap.

    Parameters
    ----------
    n_items : int
        Number of items in the bank.
    administered_items : list[int]
        Indices of items already administered in the session.
    exposure_rates : numpy.ndarray or None
        Run-level exposure-rate snapshot. ``None`` treats every item as unexposed.
    r_max : float
        Maximum exposure rate; items at or above it are only used when no other item remains.

    Returns
    -------
    numpy.ndarray
        Indices of the non-administered items below ``r_max``, or of all non-administered
        items if every one of them is at or above ``r_max``.

    Raises
    ------
    NoItemsAvailableError
        If every item has already been administered.
    """
    candidate_mask = numpy.ones(n_items, dtype=bool)
    candidate_mask[administered_items] = False
    all_candidates = numpy.flatnonzero(candidate_mask)
    if all_candidates.size == 0:
      msg = "There are no more items to apply."
      raise NoItemsAvailableError(msg)
    if exposure_rates is None:
      return all_candidates
    low_exposure = numpy.flatnonzero(candidate_mask & (exposure_rates < r_max))
    return low_exposure if low_exposure.size > 0 else all_candidates

  @staticmethod
  def _sort_by_info(item_bank: ItemBank, est_theta: float) -> list[int]:
    """Sort items by their information value for a given ability value.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    est_theta : float
        An examinee's ability.

    Returns
    -------
    list[int]
        List containing the indices of items, sorted in descending order by their
        information values at the given ability level (much like the return of
        `numpy.argsort`).
    """
    if item_bank.model == 1:
      # when the logistic model has the number of parameters <= 2,
      # all items have highest information where theta = b
      ordered_items = BaseSelector._sort_by_b(item_bank, est_theta)
    else:
      # else, sort item indexes by their information value descending and remove indexes of administered items
      ordered_items = list((-item_bank.information(est_theta)).argsort())
    return ordered_items

  @staticmethod
  def _sort_by_b(item_bank: ItemBank, est_theta: float) -> list[int]:
    """Sort items by how close their difficulty parameter is to an examinee's ability.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    est_theta : float
        An examinee's ability.

    Returns
    -------
    list[int]
        List containing the indices of items, sorted by how close their difficulty
        parameter is in relation to `est_theta` (much like the return of `numpy.argsort`).
    """
    return list(numpy.abs(item_bank.difficulty - est_theta).argsort())

  @staticmethod
  def _require_item_bank(item_bank: ItemBank | None) -> ItemBank:
    if item_bank is None:
      msg = "item_bank parameter cannot be None"
      raise ValueError(msg)
    return item_bank

  @staticmethod
  def _require_administered_items(administered_items: list[int] | None) -> list[int]:
    if administered_items is None:
      msg = "administered_items parameter cannot be None"
      raise ValueError(msg)
    return administered_items

  @staticmethod
  def _require_est_theta(est_theta: float | None) -> float:
    if est_theta is None:
      msg = "est_theta parameter cannot be None"
      raise ValueError(msg)
    return est_theta

  @abstractmethod
  def select(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,
    rng: Generator | None = None,
    exposure_rates: npt.NDArray[numpy.floating] | None = None,
    response_vector: list[bool] | None = None,
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
        Random number generator for selectors that rely on randomness.
    exposure_rates : numpy.ndarray or None, optional
        Explicit run-level exposure-rate snapshot.
    response_vector : list[bool] or None, optional
        Responses observed so far in the current session. Selectors that need
        response history can use this without relying on hidden engine state.

    Returns
    -------
    int or None
        Index of the next item to be applied, or None if there are no more items
        to be presented.
    """


class FiniteSelector(BaseSelector, ABC):
  """Base class representing a CAT item selector for fixed-length tests.

  Parameters
  ----------
  test_size : int
      Number of items to be administered in the test.
  """

  def __init__(self, test_size: int) -> None:
    """Initialize a FiniteSelector object.

    Parameters
    ----------
    test_size : int
        Number of items to be administered in the test.
    """
    self._test_size = test_size
    super().__init__()

  @property
  def test_size(self) -> int:
    """Get the number of items to be administered in the test.

    Returns
    -------
    int
        Number of items to be administered in the test.
    """
    return self._test_size
