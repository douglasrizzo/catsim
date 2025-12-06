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

  @staticmethod
  def stop_(administered_items: npt.NDArray[numpy.floating[Any]], max_items: int) -> bool:
    """Check if the maximum number of items has been reached.

    Parameters
    ----------
    administered_items : npt.NDArray[numpy.floating[Any]]
        Array of administered items.
    max_items : int
        The maximum number of items allowed.

    Returns
    -------
    bool
        True if the maximum number of items has been reached, False otherwise.

    Raises
    ------
    ValueError
        If more items than permitted were administered.
    """
    n_itens = len(administered_items)
    if n_itens > max_items:
      msg = f"More items than permitted were administered: {n_itens} > {max_items}."
      raise ValueError(msg)
    return n_itens == max_items

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

    return self.stop_(administered_items=administered_items, max_items=self._max_itens)

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

  @staticmethod
  def stop_(administered_items: npt.NDArray[numpy.floating[Any]], theta: float, min_error: float) -> bool:
    """Check if the standard error of estimation has fallen below the minimum threshold.

    Parameters
    ----------
    administered_items : npt.NDArray[numpy.floating[Any]]
        Array of administered items.
    theta : float
        The ability value to which the error will be computed.
    min_error : float
        The minimum standard error of estimation allowed.

    Returns
    -------
    bool
        True if the standard error of estimation has fallen below the minimum threshold, False otherwise.
    """
    administered_items_array = numpy.asarray(administered_items)
    see = irt.see(theta, items=administered_items_array)
    return see < min_error

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
    elif index is not None and self.simulator is not None:
      theta = self.simulator.latest_estimations[index]
      administered_items_array = self.simulator.item_bank.get_items(indices=self.simulator.administered_items[index])
    else:
      msg = (
        "Required parameters are missing. Either administered_items and theta or index and simulator must be provided."
      )
      raise ValueError(msg)
    return self.stop_(administered_items=administered_items_array, theta=theta, min_error=self._min_error)


class MaxItemMinErrorStopper(Stopper):
  """Stopping criterion combining maximum items and minimum standard error.

  The test stops when **either** the standard error of estimation falls below the
  specified threshold **or** when the maximum number of items has been administered.
  This is commonly used in variable-length CATs that have a minimum precision goal
  but also enforce a maximum test length to prevent excessively long tests.

  Parameters
  ----------
  max_itens : int
      The maximum number of items to be administered in the test.
  min_error : float
      The minimum standard error of estimation the test should achieve before stopping.
      Must be positive. Smaller values require more items for higher precision.
  """

  def __str__(self) -> str:
    """Get a string representation of the Stopper."""
    return "Maximum Item/Minimum Error Stopper"

  def __init__(self, max_itens: int, min_error: float) -> None:
    """Initialize a MaxItemMinErrorStopper.

    Parameters
    ----------
    max_itens : int
        Maximum number of items to be administered. Must be positive.
    min_error : float
        Error tolerance in estimated examinee ability to stop the test.
        The test stops when the standard error of estimation falls below this value.
    """
    super().__init__()
    self._max_itens = max_itens
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

    The test stops if either:
    1. The maximum number of items has been administered, or
    2. The standard error of estimation falls below the minimum threshold

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
        True if the test met its stopping criterion (either maximum items reached
        or standard error below minimum), False otherwise.

    Raises
    ------
    ValueError
        If more items than permitted were administered, or if required parameters are missing.
    """
    if administered_items is None and index is not None and self.simulator is not None:
      administered_items = self.simulator.item_bank.get_items(indices=self.simulator.administered_items[index])
      theta = self.simulator.latest_estimations[index]
    elif administered_items is None or theta is None:
      msg = (
        "Required parameters are missing. Either index and simulator or administered_items and theta must be provided."
      )
      raise ValueError(msg)

    return MaxItemStopper.stop_(administered_items, max_items=self._max_itens) or MinErrorStopper.stop_(
      administered_items, theta, min_error=self._min_error
    )

  @property
  def max_itens(self) -> int:
    """Get the maximum number of items the Stopper is configured to administer.

    Returns
    -------
    int
        The maximum number of items the Stopper is configured to administer.
    """
    return self._max_itens

  @property
  def min_error(self) -> float:
    """Get the minimum standard error the Stopper is configured to achieve.

    Returns
    -------
    float
        The minimum standard error the Stopper is configured to achieve.
    """
    return self._min_error
