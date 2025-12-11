"""Random selector implementations."""

from typing import Any

from ..exceptions import NoItemsAvailableError
from ..item_bank import ItemBank
from .base import BaseSelector, FiniteSelector


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
        Additional keyword arguments. Notably:

        * **rng** (:py:class:`numpy.random.Generator`) -- Random number generator used by the object,
          guarantees reproducibility of outputs.

    Returns
    -------
    int or None
        Index of the next item to be applied or `None` if there are no more items in the item bank.
    """
    item_bank, administered_items, rng = self._prepare_args(
      return_item_bank=True,
      return_administered_items=True,
      return_rng=True,
      index=index,
      item_bank=item_bank,
      administered_items=administered_items,
      **kwargs,
    )

    assert item_bank is not None
    assert administered_items is not None

    if len(administered_items) >= item_bank.n_items and not self._replace:
      msg = "There are no more items to apply."
      raise NoItemsAvailableError(msg)

    if self._replace:
      return rng.choice(item_bank.n_items)
    valid_indexes = self._get_non_administered(list(range(item_bank.n_items)), administered_items)
    return rng.choice(valid_indexes)


class The54321Selector(FiniteSelector):
  r"""Implementation of the 5-4-3-2-1 selector proposed by [McBride83]_.

  In this selector, at each step :math:`k` of a test of size :math:`K`, an item is chosen from a bin containing the
  :math:`K-k` most informative items in the bank, given the current :math:`\\hat\\theta`. As the test progresses, the
  bin gets smaller and more informative items have a higher probability of being chosen by the end of the test, when the
  estimation of ':math:`\\hat\\theta` is more precise. The 5-4-3-2-1 selector can be viewed as a specialization of the
  :py:class:`catsim.selection.RandomesqueSelector`, in which the bin size of most informative items gets smaller as the
  test progresses.
  """

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "5-4-3-2-1 Selector"

  def __init__(self, test_size: int) -> None:
    """Initialize a The54321Selector object.

    Parameters
    ----------
    test_size : int
        The number of items the test contains. The selector uses this parameter to set the bin size.
    """
    super().__init__(test_size)

  def select(
    self,
    index: int | None = None,
    item_bank: ItemBank | None = None,
    administered_items: list[int] | None = None,
    est_theta: float | None = None,
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
    est_theta : float or None, optional
        A float containing the current estimated ability. Default is None.
    **kwargs
        Additional keyword arguments. Notably:

        * **rng** (:py:class:`numpy.random.Generator`) -- Random number generator used by the object,
          guarantees reproducibility of outputs.

    Returns
    -------
    int or None
        Index of the next item to be applied or `None` if there are no more items in the item bank.
    """
    item_bank, administered_items, est_theta, rng = self._prepare_args(
      return_item_bank=True,
      return_administered_items=True,
      return_est_theta=True,
      return_rng=True,
      index=index,
      item_bank=item_bank,
      administered_items=administered_items,
      est_theta=est_theta,
      **kwargs,
    )

    assert est_theta is not None
    assert administered_items is not None
    assert item_bank is not None

    # sort item indexes by their information value descending and remove indexes of administered items
    ordered_items = self._sort_by_info(item_bank, est_theta)
    organized_items = self._get_non_administered(ordered_items, administered_items)

    if len(organized_items) == 0:
      msg = "There are no more items to apply."
      raise NoItemsAvailableError(msg)

    bin_size = self._test_size - len(administered_items)
    return rng.choice(organized_items[0:bin_size])


class RandomesqueSelector(BaseSelector):
  """Implementation of the randomesque selector proposed by [Kingsbury89]_.

  In this selector, at each step of the test, an item is randomly chosen from the :math:`n` most informative items in
  the item bank, :math:`n` being a predefined value (originally 5, but user-defined in this implementation).
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
    index: int | None = None,
    item_bank: ItemBank | None = None,
    administered_items: list[int] | None = None,
    est_theta: float | None = None,
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
    est_theta : float or None, optional
        A float containing the current estimated ability. Default is None.
    **kwargs
        Additional keyword arguments. Notably:

        * **rng** (:py:class:`numpy.random.Generator`) -- Random number generator used by the object,
          guarantees reproducibility of outputs.

    Returns
    -------
    int or None
        Index of the next item to be applied or `None` if there are no more items in the item bank.
    """
    item_bank, administered_items, est_theta, rng = self._prepare_args(
      return_item_bank=True,
      return_administered_items=True,
      return_est_theta=True,
      return_rng=True,
      index=index,
      item_bank=item_bank,
      administered_items=administered_items,
      est_theta=est_theta,
      **kwargs,
    )

    assert est_theta is not None
    assert administered_items is not None
    assert item_bank is not None

    # sort item indexes by their information value descending and remove indexes of administered items
    ordered_items = self._sort_by_info(item_bank, est_theta)
    organized_items = self._get_non_administered(ordered_items, administered_items)

    if len(organized_items) == 0:
      msg = "There are no more items to apply."
      raise NoItemsAvailableError(msg)

    return rng.choice(list(organized_items)[: self._bin_size])
