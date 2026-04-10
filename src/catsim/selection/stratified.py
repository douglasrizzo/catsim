"""Stratified selector implementations."""

from abc import abstractmethod
from typing import Any

import numpy
from numpy.typing import NDArray

from .. import irt
from ..exceptions import NoItemsAvailableError
from ..item_bank import ItemBank
from .base import FiniteSelector


class StratifiedSelector(FiniteSelector):
  """Abstract base class for finite selectors that divide the item bank into stage-aligned strata.

  Parameters
  ----------
  test_size : int
      Number of items in the test.
  sort_once : bool
      Whether the strategy allows for the item matrix to be presorted once at the
      beginning of the simulation (True) or requires resorting during the test (False).

  Notes
  -----
  For full family-level details, see :doc:`/techniques/selection/stratified-selector`.
  """

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "General Stratified Selector"

  def __init__(self, test_size: int, sort_once: bool) -> None:
    """Initialize a StratifiedSelector.

    Parameters
    ----------
    test_size : int
        Number of items in the test.
    sort_once : bool
        Whether the strategy allows for the item matrix to be presorted.
    """
    super().__init__(test_size)
    self._sort_once = sort_once
    self._presorted_items: NDArray[numpy.floating] | None = None
    self._presorted_item_bank_id: int | None = None

  @abstractmethod
  def presort_items(self, item_bank: ItemBank) -> NDArray[numpy.floating]:
    """Presort the item matrix according to the strategy employed by this selector.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.

    Returns
    -------
    numpy.ndarray
        Array of item indices sorted according to the strategy.
    """

  def postsort_items(self, item_bank: ItemBank, **kwargs: Any) -> NDArray[numpy.floating]:  # noqa: ARG002
    """Sort the item matrix before selecting each new item.

    This default implementation simply returns the presorted items, or sorts them using
    the :py:func:`presort_items` method and returns them.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    numpy.ndarray
        Array of item indices sorted according to the strategy.
    """
    return self._get_presorted_items(item_bank)

  def _get_presorted_items(self, item_bank: ItemBank) -> NDArray[numpy.floating]:
    if self._presorted_items is None or self._presorted_item_bank_id != id(item_bank):
      self._presorted_items = self.presort_items(item_bank)
      self._presorted_item_bank_id = id(item_bank)
    return self._presorted_items

  def select(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,
    rng: numpy.random.Generator | None = None,  # noqa: ARG002
    exposure_rates: NDArray[numpy.floating] | None = None,  # noqa: ARG002
  ) -> int | None:
    """Return the index of the next item to be administered.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    administered_items : list[int]
        A list containing the indexes of items that were already administered.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    int or None
        Index of the next item to be applied or `None` if there are no more strata to get items from.
    """
    # divide the item matrix into strata and get the stratum in which the examinee is
    stratum_index = len(administered_items)
    slices, pointer, max_pointer = self._get_stratum(item_bank, stratum_index)

    if self._sort_once:
      sorted_items = self._get_presorted_items(item_bank)
    else:
      sorted_items = self.postsort_items(item_bank, est_theta=est_theta)

    # if the selected item has already been administered, select the next one
    while sorted_items[pointer] in administered_items:
      pointer += 1
      if pointer == max_pointer:
        msg = f"There are no more items to be selected from stratum {slices[len(administered_items)]}"
        raise NoItemsAvailableError(msg)

    return sorted_items[pointer]

  def _get_stratum(self, item_bank: ItemBank, stratum_index: int) -> tuple[NDArray[numpy.floating], int, int]:
    slices = numpy.linspace(0, item_bank.n_items, self._test_size, endpoint=False, dtype="i")
    if stratum_index >= len(slices):
      msg = (
        f"{self}: test size is larger than was informed to the selector\n"
        f"Length of administered items:\t{stratum_index}\n"
        f"Total length of the test:\t{self._test_size}\n"
        f"Number of slices:\t{len(slices)}"
      )
      raise RuntimeError(msg)
    pointer = slices[stratum_index]
    max_pointer = item_bank.n_items if stratum_index == self._test_size - 1 else slices[stratum_index + 1]

    return slices, pointer, max_pointer


class AStratSelector(StratifiedSelector):
  r"""Implementation of the :math:`\alpha`-stratified selector.

  Parameters
  ----------
  test_size : int
      The number of items the test contains. The selector uses this parameter to create
      the correct number of strata.

  Notes
  -----
  For full algorithmic details, see :doc:`/techniques/selection/a-strat-selector`.
  """

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "a-Stratified Selector"

  def __init__(self, test_size: int) -> None:
    """Initialize an AStratSelector object.

    Parameters
    ----------
    test_size : int
        Number of items the test contains.
    """
    super().__init__(test_size, True)

  def presort_items(self, item_bank: ItemBank) -> NDArray[numpy.floating]:  # noqa: PLR6301
    """Presort the item matrix in ascending order according to the discrimination of each item.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.

    Returns
    -------
    numpy.ndarray
        Array of item indices sorted in ascending order by discrimination (a parameter).
    """
    return item_bank.discrimination.argsort()


class AStratBBlockSelector(StratifiedSelector):
  r"""Implementation of the :math:`\alpha`-stratified selector with :math:`b` blocking.

  Parameters
  ----------
  test_size : int
      The number of items the test contains. The selector uses this parameter to
      create the correct number of strata.

  Notes
  -----
  For full algorithmic details, see :doc:`/techniques/selection/a-strat-b-block-selector`.
  """

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "a-Stratified b-Blocking Selector"

  def __init__(self, test_size: int) -> None:
    """Initialize a AStratBBlockSelector object.

    Parameters
    ----------
    test_size : int
        Number of items the test contains.
    """
    super().__init__(test_size, True)

  def presort_items(self, item_bank: ItemBank) -> NDArray[numpy.floating]:
    """Presort items in ascending order of discrimination each item, then each strata according to item difficulty.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.

    Returns
    -------
    numpy.ndarray
        The sorted item matrix.
    """
    # sort items by their b values, in ascending order
    presorted_items = item_bank.difficulty.argsort()

    final_indices = []
    for stratum_index in range(self._test_size):
      _slices, pointer, max_pointer = self._get_stratum(item_bank, stratum_index)
      indices_current_stratum = presorted_items[pointer:max_pointer]
      items_current_stratum = item_bank.get_items(indices_current_stratum)
      sorted_indices_current_stratum = items_current_stratum[:, 0].argsort()
      # sort the items in the current stratum by their discrimination values, in ascending order
      global_sorted_indices_current_stratum = indices_current_stratum[sorted_indices_current_stratum]
      final_indices.extend(global_sorted_indices_current_stratum)

    # sort the item bank first by the items maximum information, ascending
    # then by their information to the examinee's cuirrent theta, descending
    return numpy.array(final_indices)


class MaxInfoStratSelector(StratifiedSelector):
  """Implementation of the maximum information stratification (MIS) selector.

  Parameters
  ----------
  test_size : int
      The number of items the test contains. The selector uses this parameter to create
      the correct number of strata.

  Notes
  -----
  For full algorithmic details, see :doc:`/techniques/selection/max-info-strat-selector`.
  """

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "Maximum Information Stratification Selector"

  def __init__(self, test_size: int) -> None:
    """Initialize a MaxInfoStratSelector object.

    Parameters
    ----------
    test_size : int
        Number of items the test contains.
    """
    super().__init__(test_size, False)

  def presort_items(self, item_bank: ItemBank) -> NDArray[numpy.floating]:  # noqa: PLR6301
    """Presort items in ascending order of maximum information.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.

    Returns
    -------
    numpy.ndarray
        The sorted item matrix.
    """
    # Use cached max info values from ItemBank - this is a key optimization!
    # get the theta values in which items are maximally informative
    # get the information values for all items at their maximum points
    item_maxinfo = item_bank.max_info_values  # Using cached values!
    # globally sort item bank by item max information
    return item_maxinfo.argsort()

  def postsort_items(
    self,
    item_bank: ItemBank,
    est_theta: float,
    **kwargs: Any,  # noqa: ARG002
  ) -> NDArray[numpy.floating]:
    """Divide the item bank into strata and sort each one in descending order of information for the current theta.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    est_theta : float
        The current estimate of the examinee's ability.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    numpy.ndarray
        The sorted item matrix.
    """
    # recover items presorted by the first rule
    presorted_items = self._get_presorted_items(item_bank)
    # run through each stratum and sort items in descending order according to
    # their information for the current theta value
    final_indices = []
    for stratum_index in range(self._test_size):
      # grab stratum pointers
      _slices, pointer, max_pointer = self._get_stratum(item_bank, stratum_index)
      item_indices_current_stratum = presorted_items[pointer:max_pointer]  # item indices for the current stratum
      items_current_stratum: NDArray[numpy.floating] = item_bank.get_items(
        item_indices_current_stratum
      )  # item params for the current stratum
      # their information for this theta
      info_items_current_stratum_current_theta: NDArray[numpy.floating] = irt.inf_hpc(est_theta, items_current_stratum)
      item_indices_current_stratum_sorted_by_info = item_indices_current_stratum[
        (-info_items_current_stratum_current_theta).argsort()
      ]
      final_indices.extend(item_indices_current_stratum_sorted_by_info)

    # sort the item bank first by the items maximum information, ascending
    # then by their information to the examinee's cuirrent theta, descending
    return numpy.array(final_indices)


class MaxInfoBBlockSelector(MaxInfoStratSelector):
  """Implementation of the maximum information stratification with :math:`b` blocking (MIS-B) selector.

  Parameters
  ----------
  test_size : int
      The number of items the test contains. The selector uses this parameter to
      create the correct number of strata.

  Notes
  -----
  For full algorithmic details, see :doc:`/techniques/selection/max-info-b-block-selector`.
  """

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "Maximum Information Stratification with b-Blocking Selector"

  def presort_items(self, item_bank: ItemBank) -> NDArray[numpy.floating]:
    """Presort the item matrix according to the information of each item at their maximum.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.

    Returns
    -------
    numpy.ndarray
        The sorted item matrix.
    """
    # Use cached max info values from ItemBank - key optimization!
    # get the theta values in which items are maximally informative
    theta_maxinfo = item_bank.max_info_thetas  # Using cached values!
    # sort items by theta
    presorted_items = theta_maxinfo.argsort()
    # get the information values for all items at their maximum points
    item_maxinfo = item_bank.max_info_values  # Using cached values!

    final_indices = []
    for stratum_index in range(self._test_size):
      _slices, pointer, max_pointer = self._get_stratum(item_bank, stratum_index)
      indices_current_stratum = presorted_items[pointer:max_pointer]
      # sort items in the current stratum by maximum information, in ascending order
      sorted_indices_current_stratum = item_maxinfo[indices_current_stratum].argsort()
      global_sorted_indices_current_stratum = indices_current_stratum[sorted_indices_current_stratum]
      final_indices.extend(global_sorted_indices_current_stratum)

    # sanity check to make sure all indices are present and unique
    if len(final_indices) != len(set(final_indices)):
      msg = "Presorted indices must be unique"
      raise RuntimeError(msg)
    return numpy.array(final_indices)
