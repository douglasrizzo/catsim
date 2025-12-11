"""Concrete stopping criteria implementations."""

from typing import Any

import numpy
import numpy.typing as npt

from .. import irt
from ..item_bank import ItemBank
from .base import BaseStopper


class TestLengthStopper(BaseStopper):
  """Base class for stoppers with common min/max item constraints and bank exhaustion checks.

  This class provides common functionality for stopping criteria including:
  - Minimum items requirement (test cannot stop before min_items are administered)
  - Maximum items constraint (test must stop when max_items are reached)
  - Item bank exhaustion detection (test stops when all items are used)

  Subclasses must implement the `_check_stopping_criterion` method to define
  their specific stopping logic.

  Parameters
  ----------
  min_items : int or None, optional
      Minimum number of items that must be administered before the test can stop.
      If None, no minimum is enforced. Default is None.
  max_items : int or None, optional
      Maximum number of items that can be administered. Test stops when this limit
      is reached regardless of other criteria. If None, no maximum is enforced.
      Default is None.

  Notes
  -----
  The stopping logic follows this priority order:
  1. Stop if max_items reached (hard stop)
  2. Stop if item bank exhausted (hard stop)
  3. Do not stop if min_items not yet reached (regardless of other criteria)
  4. Check the specific stopping criterion implemented by the subclass
  """

  def __init__(self, min_items: int | None = None, max_items: int | None = None) -> None:
    """Initialize a BaseStopper with optional min/max item constraints.

    Parameters
    ----------
    min_items : int or None, optional
        Minimum number of items before test can stop. Must be positive if provided.
        Default is None.
    max_items : int or None, optional
        Maximum number of items before test must stop. Must be positive if provided.
        Default is None.

    Raises
    ------
    ValueError
        If min_items or max_items are not positive, or if min_items > max_items.
    """
    super().__init__()

    if min_items is not None and min_items < 1:
      msg = f"min_items must be positive, got {min_items}"
      raise ValueError(msg)

    if max_items is not None and max_items < 1:
      msg = f"max_items must be positive, got {max_items}"
      raise ValueError(msg)

    if min_items is not None and max_items is not None and min_items > max_items:
      msg = f"min_items ({min_items}) cannot be greater than max_items ({max_items})"
      raise ValueError(msg)

    self._min_items = min_items
    self._max_items = max_items

  def stop(
    self,
    index: int | None = None,
    _item_bank: ItemBank | None = None,
    administered_items: npt.NDArray[numpy.floating[Any]] | None = None,
    theta: float | None = None,
    **kwargs: Any,
  ) -> bool:
    """Check whether the test should stop based on common constraints and specific criterion.

    Parameters
    ----------
    index : int or None, optional
        The index of the current examinee. Default is None.
    _item_bank : ItemBank or None, optional
        The item bank being used. Default is None.
    administered_items : npt.NDArray[numpy.floating[Any]] or None, optional
        A matrix containing the parameters of items that were already administered.
        Default is None.
    theta : float or None, optional
        An ability value. Default is None.
    **kwargs : dict
        Additional keyword arguments passed to the specific stopping criterion.

    Returns
    -------
    bool
        True if the test should stop, False otherwise.

    Raises
    ------
    ValueError
        If required parameters are missing.
    """
    # Extract data from simulator if not provided directly
    if administered_items is not None:
      n_items = len(administered_items)
      administered_items_array = numpy.asarray(administered_items)
    elif index is not None and self._simulator is not None:
      n_items = len(self.simulator.administered_items[index])
      administered_items_array = self.simulator.item_bank.get_items(indices=self.simulator.administered_items[index])
      if theta is None:
        theta = self.simulator.estimations[index][-1]
      if _item_bank is None:
        _item_bank = self.simulator.item_bank
    else:
      msg = "Required parameters are missing. Either administered_items or index and simulator must be provided."
      raise ValueError(msg)

    # Hard stop: max_items reached
    if self._max_items is not None and n_items >= self._max_items:
      return True

    # Hard stop: item bank exhausted
    if _item_bank is not None and n_items >= len(_item_bank):
      return True

    # Cannot stop yet: min_items not reached
    if self._min_items is not None and n_items < self._min_items:
      return False

    # Check the specific stopping criterion
    return self._check_stopping_criterion(administered_items_array, theta, **kwargs)

  def _check_stopping_criterion(  # noqa: PLR6301
    self,
    administered_items: npt.NDArray[numpy.floating[Any]],  # noqa: ARG002
    theta: float | None,  # noqa: ARG002
    **kwargs: Any,  # noqa: ARG002
  ) -> bool:
    """Check the specific stopping criterion implemented by the subclass.

    This method must be implemented by subclasses to define their specific
    stopping logic (e.g., error threshold, confidence interval).

    Parameters
    ----------
    administered_items : npt.NDArray[numpy.floating[Any]]
        Array containing the parameters of administered items.
    theta : float or None
        Current ability estimate.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    bool
        True if the specific stopping criterion is met, False otherwise.
    """
    return False

  @property
  def min_items(self) -> int | None:
    """Get the minimum number of items.

    Returns
    -------
    int or None
        The minimum number of items, or None if no minimum is set.
    """
    return self._min_items

  @property
  def max_items(self) -> int | None:
    """Get the maximum number of items.

    Returns
    -------
    int or None
        The maximum number of items, or None if no maximum is set.
    """
    return self._max_items


class MinErrorStopper(TestLengthStopper):
  """Stopping criterion based on minimum standard error of estimation.

  The test stops when the standard error of estimation (see :py:func:`catsim.irt.see`)
  falls below the specified threshold. This is commonly used in variable-length CATs
  to achieve a desired level of measurement precision.

  This stopper also enforces optional minimum/maximum item constraints and stops
  when the item bank is exhausted (via :py:class:`BaseStopper`).

  Parameters
  ----------
  min_error : float
      The minimum standard error of estimation the test must achieve before stopping.
      Must be positive. Smaller values require more items for higher precision.
  min_items : int or None, optional
      Minimum number of items that must be administered before the test can stop.
      Default is None (no minimum).
  max_items : int or None, optional
      Maximum number of items that can be administered. Default is None (no maximum).

  Examples
  --------
  >>> # Stop when error < 0.3
  >>> stopper = MinErrorStopper(0.3)

  >>> # Stop when error < 0.3, but only after at least 10 items
  >>> stopper = MinErrorStopper(0.3, min_items=10)

  >>> # Stop when error < 0.3, or when 50 items reached
  >>> stopper = MinErrorStopper(0.3, max_items=50)

  >>> # Stop when error < 0.3, between 10 and 50 items
  >>> stopper = MinErrorStopper(0.3, min_items=10, max_items=50)
  """

  def __str__(self) -> str:
    """Get a string representation of the Stopper."""
    parts = [f"min_error={self._min_error}"]
    if self._min_items is not None:
      parts.append(f"min_items={self._min_items}")
    if self._max_items is not None:
      parts.append(f"max_items={self._max_items}")
    return f"MinErrorStopper({', '.join(parts)})"

  def __init__(self, min_error: float, min_items: int | None = None, max_items: int | None = None) -> None:
    """Initialize a MinErrorStopper.

    Parameters
    ----------
    min_error : float
        Error tolerance in estimated examinee ability to stop the test.
        The test stops when the standard error of estimation falls below this value.
        Must be positive.
    min_items : int or None, optional
        Minimum number of items before test can stop. Default is None.
    max_items : int or None, optional
        Maximum number of items before test must stop. Default is None.

    Raises
    ------
    ValueError
        If min_error is not positive, or if min/max_items constraints are invalid.
    """
    if min_error <= 0:
      msg = f"min_error must be positive, got {min_error}"
      raise ValueError(msg)

    super().__init__(min_items=min_items, max_items=max_items)
    self._min_error = min_error

  def _check_stopping_criterion(
    self,
    administered_items: npt.NDArray[numpy.floating[Any]],
    theta: float | None,
    **kwargs: Any,  # noqa: ARG002
  ) -> bool:
    """Check if the standard error is below the threshold.

    Parameters
    ----------
    administered_items : npt.NDArray[numpy.floating[Any]]
        Array of administered item parameters.
    theta : float or None
        Current ability estimate.
    **kwargs : dict
        Additional keyword arguments (unused).

    Returns
    -------
    bool
        True if standard error is below min_error, False otherwise.

    Raises
    ------
    ValueError
        If theta is None.
    """
    if theta is None:
      msg = "theta is required for MinErrorStopper"
      raise ValueError(msg)

    if len(administered_items) == 0:
      return False

    see = irt.see(theta, administered_items)
    return see < self._min_error

  @property
  def min_error(self) -> float:
    """Get the minimum error threshold.

    Returns
    -------
    float
        The minimum error threshold.
    """
    return self._min_error


class ConfidenceIntervalStopper(TestLengthStopper):
  r"""Stopping criterion based on confidence interval falling within discrete ability intervals.

  This stopper is designed for tests with discrete performance levels (e.g., letter grades
  A, B, C, D, F) defined by intervals on the ability scale. The test stops when the
  confidence interval for the examinee's ability estimate falls entirely within one of
  these discrete intervals, indicating sufficient precision to classify the examinee.

  This stopper also enforces optional minimum/maximum item constraints and stops
  when the item bank is exhausted (via :py:class:`BaseStopper`).

  Parameters
  ----------
  interval_bounds : list[float]
      Sorted list of boundary points defining the discrete intervals on the ability scale.
      For example, [-2.0, -0.5, 0.5, 2.0] defines 5 intervals:
      (-inf, -2.0), [-2.0, -0.5), [-0.5, 0.5), [0.5, 2.0), [2.0, inf)
  confidence : float, optional
      The confidence level for computing the confidence interval, must be between 0 and 1.
      Default is 0.95 (95% confidence).
  min_items : int or None, optional
      Minimum number of items that must be administered before the test can stop.
      Default is None (no minimum).
  max_items : int or None, optional
      Maximum number of items that can be administered. Default is None (no maximum).

  Examples
  --------
  >>> # Define grade boundaries: F (<-1), D [-1, 0), C [0, 1), B [1, 2), A (>=2)
  >>> stopper = ConfidenceIntervalStopper([-1.0, 0.0, 1.0, 2.0], confidence=0.95)
  >>> # Test stops when 95% CI is entirely within one grade interval

  >>> # With minimum items constraint
  >>> stopper = ConfidenceIntervalStopper(
  ...     [-1.0, 0.0, 1.0, 2.0], confidence=0.90, min_items=10
  ... )
  >>> # Test cannot stop before 10 items, even if CI criterion is met

  >>> # With maximum items constraint
  >>> stopper = ConfidenceIntervalStopper(
  ...     [-1.0, 0.0, 1.0, 2.0], confidence=0.95, max_items=50
  ... )
  >>> # Test stops at 50 items even if CI criterion is not met
  """

  def __str__(self) -> str:
    """Get a string representation of the Stopper."""
    parts = [f"confidence={self._confidence}"]
    if self._min_items is not None:
      parts.append(f"min_items={self._min_items}")
    if self._max_items is not None:
      parts.append(f"max_items={self._max_items}")
    return f"ConfidenceIntervalStopper({', '.join(parts)})"

  def __init__(
    self,
    interval_bounds: list[float],
    confidence: float = 0.95,
    min_items: int | None = None,
    max_items: int | None = None,
  ) -> None:
    """Initialize a ConfidenceIntervalStopper.

    Parameters
    ----------
    interval_bounds : list[float]
        Sorted list of boundary points on the ability scale. Must contain at least one value.
    confidence : float, optional
        Confidence level for the interval computation (between 0 and 1). Default is 0.95.
    min_items : int or None, optional
        Minimum number of items before test can stop. Default is None.
    max_items : int or None, optional
        Maximum number of items before test must stop. Default is None.

    Raises
    ------
    ValueError
        If interval_bounds is empty, not sorted, confidence is not between 0 and 1,
        or if min/max_items constraints are invalid.
    """
    if not interval_bounds:
      msg = "interval_bounds must contain at least one value"
      raise ValueError(msg)

    # Verify bounds are sorted
    sorted_bounds = sorted(interval_bounds)
    if sorted_bounds != list(interval_bounds):
      msg = "interval_bounds must be sorted in ascending order"
      raise ValueError(msg)

    if not 0 < confidence < 1:
      msg = f"confidence must be between 0 and 1, got {confidence}"
      raise ValueError(msg)

    super().__init__(min_items=min_items, max_items=max_items)
    self._interval_bounds = list(interval_bounds)
    self._confidence = confidence

  def _check_stopping_criterion(
    self,
    administered_items: npt.NDArray[numpy.floating[Any]],
    theta: float | None,
    **kwargs: Any,  # noqa: ARG002
  ) -> bool:
    """Check if the confidence interval falls entirely within a discrete interval.

    Parameters
    ----------
    administered_items : npt.NDArray[numpy.floating[Any]]
        Array of administered item parameters.
    theta : float or None
        Current ability estimate.
    **kwargs : dict
        Additional keyword arguments (unused).

    Returns
    -------
    bool
        True if the confidence interval falls entirely within one of the discrete
        intervals, False otherwise.

    Raises
    ------
    ValueError
        If theta is None.
    """
    if theta is None:
      msg = "theta is required for ConfidenceIntervalStopper"
      raise ValueError(msg)

    # Need at least one item to compute confidence interval
    if len(administered_items) == 0:
      return False

    # Compute confidence interval
    lower_bound, upper_bound = irt.confidence_interval(theta, administered_items, self._confidence)

    # Check if the entire confidence interval falls within one of the discrete intervals
    # The discrete intervals are:
    # (-inf, bounds[0]), [bounds[0], bounds[1]), ..., [bounds[n-1], inf)

    # Check if CI is in the leftmost interval (-inf, bounds[0])
    if upper_bound <= self._interval_bounds[0]:
      return True

    # Check if CI is in the rightmost interval [bounds[-1], inf)
    if lower_bound >= self._interval_bounds[-1]:
      return True

    # Check if CI falls within any middle interval [bounds[i], bounds[i+1])
    for i in range(len(self._interval_bounds) - 1):
      if lower_bound >= self._interval_bounds[i] and upper_bound <= self._interval_bounds[i + 1]:
        return True

    return False

  @property
  def interval_bounds(self) -> list[float]:
    """Get the interval boundary points.

    Returns
    -------
    list[float]
        The boundary points defining the discrete intervals.
    """
    return self._interval_bounds

  @property
  def confidence(self) -> float:
    """Get the confidence level.

    Returns
    -------
    float
        The confidence level used for computing confidence intervals.
    """
    return self._confidence
