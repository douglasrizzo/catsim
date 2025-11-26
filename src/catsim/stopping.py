from typing import Any

import numpy
import numpy.typing as npt

from . import irt
from .item_bank import ItemBank
from .simulation import Stopper


class MaxItemStopper(Stopper):
  """Stopping criterion based on maximum number of items in a test.

  The test stops when the specified maximum number of items has been administered.
  This is the most common stopping criterion in fixed-length CATs.

  Parameters
  ----------
  max_itens : int
      The maximum number of items to be administered in the test.
  """

  def __str__(self) -> str:
    """Get a string representation of the Stopper."""
    return "Maximum Item Number Initializer"

  def __init__(self, max_itens: int) -> None:
    """Initialize a MaxItemStopper.

    Parameters
    ----------
    max_itens : int
        Maximum number of items to be administered. Must be positive.
    """
    super().__init__()
    self._max_itens = max_itens

  def stop(
    self,
    index: int | None = None,
    _item_bank: ItemBank | None = None,
    administered_items: npt.NDArray[numpy.floating[Any]] | None = None,
    **kwargs: Any,  # noqa: ARG002
  ) -> bool:
    """Check whether the test reached its stopping criterion for the given user.

    Parameters
    ----------
    index : int or None, optional
        The index of the current examinee. Default is None.
    administered_items : npt.NDArray[numpy.floating[Any]] or None, optional
        A matrix containing the parameters of items that were already administered.
        Default is None.
    **kwargs : dict
        Additional keyword arguments. Not used by this method.

    Returns
    -------
    bool
        True if the test met its stopping criterion (maximum items reached), False otherwise.

    Raises
    ------
    ValueError
        If more items than permitted were administered, or if required parameters are missing.
    """
    if administered_items is None and index is not None and self.simulator is not None:
      administered_items = self.simulator.item_bank.get_items(self.simulator.administered_items[index])
    elif administered_items is None:
      msg = "Required parameters are missing. Either index and simulator or administered_items must be provided."
      raise ValueError(msg)

    n_itens = len(administered_items)
    if n_itens > self._max_itens:
      msg = f"More items than permitted were administered: {n_itens} > {self._max_itens}."
      raise ValueError(msg)

    return n_itens == self._max_itens

  @property
  def max_itens(self) -> int:
    """Get the maximum number of items the Stopper is configured to administer.

    Returns
    -------
    int
        The maximum number of items the Stopper is configured to administer.
    """
    return self._max_itens


class MinErrorStopper(Stopper):
  """Stopping criterion based on minimum standard error of estimation.

  The test stops when the standard error of estimation (see :py:func:`catsim.irt.see`)
  falls below the specified threshold. This is commonly used in variable-length CATs
  to achieve a desired level of measurement precision.

  Parameters
  ----------
  min_error : float
      The minimum standard error of estimation the test must achieve before stopping.
      Must be positive. Smaller values require more items for higher precision.
  """

  def __str__(self) -> str:
    """Get a string representation of the Stopper."""
    return "Minimum Error Initializer"

  def __init__(self, min_error: float) -> None:
    """Initialize a MinErrorStopper.

    Parameters
    ----------
    min_error : float
        Error tolerance in estimated examinee ability to stop the test.
        The test stops when the standard error of estimation falls below this value.
    """
    super().__init__()
    self._min_error = min_error

  def stop(
    self,
    index: int | None = None,
    _item_bank: ItemBank | None = None,
    administered_items: npt.NDArray[numpy.floating[Any]] | None = None,
    theta: float | None = None,
    **kwargs: Any,  # noqa: ARG002
  ) -> bool:
    """Check whether the test reached its stopping criterion.

    Parameters
    ----------
    index : int or None, optional
        The index of the current examinee. Default is None.
    administered_items : npt.NDArray[numpy.floating[Any]] or None, optional
        A matrix containing the parameters of items that were already administered.
        Default is None.
    theta : float or None, optional
        An ability value to which the error will be computed. Default is None.
    **kwargs : dict
        Additional keyword arguments. Not used by this method.

    Returns
    -------
    bool
        True if the test met its stopping criterion (standard error below minimum),
        False otherwise.

    Raises
    ------
    ValueError
        If required parameters are missing.
    """
    if administered_items is not None and theta is not None:
      administered_items_array = numpy.asarray(administered_items)
      see = irt.see(theta, administered_items_array)
    elif index is not None and self.simulator is not None:
      theta = self.simulator.latest_estimations[index]
      administered_items_array = self.simulator.item_bank.get_items(self.simulator.administered_items[index])
      see = irt.see(theta, administered_items_array)
    else:
      msg = (
        "Required parameters are missing. Either administered_items and theta or index and simulator must be provided."
      )
      raise ValueError(msg)
    return see < self._min_error
