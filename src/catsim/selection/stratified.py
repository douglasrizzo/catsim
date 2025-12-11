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
  """Abstract class for stratified finite item selection strategies.

  Stratified selectors divide the item bank into strata and select items from different
  strata as the test progresses, helping to balance item exposure and test characteristics.

  Parameters
  ----------
  test_size : int
      Number of items in the test.
  sort_once : bool
      Whether the strategy allows for the item matrix to be presorted once at the
      beginning of the simulation (True) or requires resorting during the test (False).
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

  def postsort_items(
    self,
    item_bank: ItemBank,
    using_simulator_props: bool,
    **kwargs: Any,  # noqa: ARG002
  ) -> NDArray[numpy.floating]:
    """Sort the item matrix before selecting each new item.

    This default implementation simply returns the presorted items, or sorts them using
    the :py:func:`presort_items` method and returns them.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    using_simulator_props : bool
        Whether the selector is being executed inside a Simulator.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    numpy.ndarray
        Array of item indices sorted according to the strategy.
    """
    if using_simulator_props:
      return self._presorted_items
    return self.presort_items(item_bank)

  def preprocess(self) -> None:  # noqa: D102
    self._presorted_items = self.presort_items(self.simulator.item_bank)

  def select(
    self,
    index: int | None = None,
    item_bank: ItemBank | None = None,
    administered_items: list[int] | None = None,
    **kwargs: Any,
  ) -> int | None:
    """Return the index of the next item to be administered.

    Parameters
    ----------
    index : int or None, optional
        The index of the current examinee in the simulator. Default is None.
    item_bank : ItemBank or None, optional
        An ItemBank containing item parameters. Default is None.
    administered_items : list[int] or None, optional
        A list containing the indexes of items that were already administered. Default is None.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    int or None
        Index of the next item to be applied or `None` if there are no more strata to get items from.
    """
    item_bank, administered_items, est_theta = self._prepare_args(
      return_item_bank=True,
      return_administered_items=True,
      return_est_theta=True,
      index=index,
      item_bank=item_bank,
      administered_items=administered_items,
      **kwargs,
    )

    assert item_bank is not None
    assert administered_items is not None
    assert est_theta is not None

    # divide the item matrix into strata and get the stratum in which the examinee is
    stratum_index = len(administered_items)
    try:
      slices, pointer, max_pointer = self._get_stratum(item_bank, stratum_index)
    except IndexError as ierr:
      msg = (
        f"{self}: test size is larger than was informed to the selector\n"
        f"Length of administered items:\t{len(administered_items)}\n"
        f"Total length of the test:\t{self._test_size}\n"
        f"Number of slices:\t{len(slices)}"
      )
      raise RuntimeError(msg) from ierr

    using_simulator_props = index is not None

    if using_simulator_props and self._sort_once:
      # if running through a simulator and the selector allows presorting, get the presorted item matrix
      sorted_items = self._presorted_items
    else:
      # allow the selector to resort the item matrix at this point in the test
      kwargs["using_simulator_props"] = using_simulator_props
      sorted_items = self.postsort_items(item_bank, using_simulator_props, est_theta=est_theta)

    # if the selected item has already been administered, select the next one
    while sorted_items[pointer] in administered_items:
      pointer += 1
      if pointer == max_pointer:
        msg = f"There are no more items to be selected from stratum {slices[len(administered_items)]}"
        raise NoItemsAvailableError(msg)

    return sorted_items[pointer]

  def _get_stratum(self, item_bank: ItemBank, stratum_index: int) -> tuple[NDArray[numpy.floating], int, int]:
    slices = numpy.linspace(0, item_bank.n_items, self._test_size, endpoint=False, dtype="i")
    pointer = slices[stratum_index]
    max_pointer = item_bank.n_items if stratum_index == self._test_size - 1 else slices[stratum_index + 1]

    return slices, pointer, max_pointer


class AStratSelector(StratifiedSelector):
  r"""Implementation of the :math:`\alpha`-stratified selector proposed by [Chang99]_.

  In this selector, the item bank is sorted in ascending order according to the items'
  discrimination parameter and then separated into :math:`K` strata (:math:`K` being the
  test size), each stratum containing gradually higher average discrimination. The
  :math:`\alpha`-stratified selector then selects the first non-administered item from
  stratum :math:`k`, where :math:`k` represents the position in the test of the current
  item the examinee is being presented.

  This method helps control item exposure by ensuring items with different discrimination
  levels are distributed throughout the test.

  .. image:: ../sphinx/alpha-strat.*

  Parameters
  ----------
  test_size : int
      The number of items the test contains. The selector uses this parameter to create
      the correct number of strata.
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
  r"""Implementation of the :math:`\\alpha`-stratified selector with :math:`b` blocking proposed by [Chang2001]_.

  In this selector, the item bank is sorted in ascending order according to the items difficulty parameter and then
  separated into :math:`M` strata, each stratum containing gradually higher average difficulty.

  Each of the :math:`M` strata is then again separated into :math:`K` sub-strata (:math:`k` being the test size),
  according to their discrimination. The final item bank is then ordered such that the first sub-strata of each strata
  forms the first strata of the new ordered item bank, and so on. This method tries to balance the distribution of both
  parameters between all strata, after perceiving that they are correlated.

  .. image:: ../sphinx/b-blocking.*

  Parameters
  ----------
  test_size : int
      The number of items the test contains. The selector uses this parameter to
      create the correct number of strata.
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
  """Implementation of the maximum information stratification (MIS) selector proposed by [Bar06]_.

  In this selector, the item bank is sorted in ascending order according to the items'
  maximum information and then separated into :math:`K` strata (:math:`K` being the test
  size), each stratum containing items with gradually higher maximum information. The MIS
  selector then selects the first non-administered item from stratum :math:`k`, where
  :math:`k` represents the position in the test of the current item the examinee is being
  presented.

  .. image:: ../sphinx/mis.*

  This method claims to work better than the :math:`a`-stratified method by [Chang99]_ for
  the three-parameter logistic model of IRT, since item difficulty and maximum information
  are not positioned in the same place in the ability scale in 3PL.

  Parameters
  ----------
  test_size : int
      The number of items the test contains. The selector uses this parameter to create
      the correct number of strata.
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
    using_simulator_props: bool,
    est_theta: float,
    **kwargs: Any,  # noqa: ARG002
  ) -> NDArray[numpy.floating]:
    """Divide the item bank into strata and sort each one in descending order of information for the current theta.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    using_simulator_props : bool
        Whether the selector is being executed inside a Simulator.
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
    presorted_items = self._presorted_items if using_simulator_props else self.presort_items(item_bank)
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
  """Implementation of the maximum information stratification with :math:`b` blocking (MIS-B) selector [Bar06]_.

  In this selector, the item bank is sorted in ascending order according to the items difficulty parameter and then
  separated into :math:`M` strata, each stratum containing gradually higher average difficulty.

  Each of the :math:`M` strata is then again separated into :math:`K` sub-strata (:math:`k` being the test size),
  according to the items maximum information. The final item bank is then ordered such that the first sub-strata of each
  strata forms the first strata of the new ordered item bank, and so on. This method tries to balance the distribution
  of both parameters between all strata and works better than the :math:`a`-stratified with :math:`b` blocking method by
  [Chang2001]_ for the three-parameter logistic model of IRT, since item difficulty and maximum information are not
  positioned in the same place in the ability scale in 3PL. This may also apply, although not mentioned by the authors,
  for the 4PL.

  .. image:: ../sphinx/mis-b.*

  Parameters
  ----------
  test_size : int
      The number of items the test contains. The selector uses this parameter to
      create the correct number of strata.
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
    assert len(final_indices) == len(set(final_indices))
    return numpy.array(final_indices)
