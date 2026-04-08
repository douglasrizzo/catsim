"""Base class for CAT selectors."""

from abc import ABC, abstractmethod
from typing import Any

import numpy

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
    est_theta: float | None = None,
    **kwargs: Any,
  ) -> int | None:
    """Return the index of the next item to be administered.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    administered_items : list[int]
        A list containing the indexes of items that were already administered.
    est_theta : float or None, optional
        A float containing the current estimated ability. Default is None.
    **kwargs : dict
        Additional keyword arguments such as a random number generator or
        exposure-rate snapshot.

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
    self._overlap_rate: float | None = None
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

  @property
  def overlap_rate(self) -> float | None:
    """Get the overlap rate of the test, if it is of finite length.

    Returns
    -------
    float or None
        Overlap rate of the test, or None if not yet computed.
    """
    return self._overlap_rate
