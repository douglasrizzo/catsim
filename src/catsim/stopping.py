from enum import Enum
from typing import Any

import numpy
import numpy.typing as npt

from . import irt
from .item_bank import ItemBank
from .simulation import Stopper


class CombinationStrategy(Enum):
  """Strategy for combining multiple stopping criteria.

  Attributes
  ----------
  OR : str
      Test stops when ANY of the stoppers indicates stopping (logical OR).
      This is useful when you want the test to end as soon as any condition is met.
  AND : str
      Test stops when ALL of the stoppers indicate stopping (logical AND).
      This is useful when you want the test to continue until all conditions are met.
  """

  OR = "or"
  AND = "and"


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
    elif index is not None and self.simulator is not None:
      theta = self.simulator.latest_estimations[index]
      administered_items_array = self.simulator.item_bank.get_items(indices=self.simulator.administered_items[index])
    else:
      msg = (
        "Required parameters are missing. Either administered_items and theta or index and simulator must be provided."
      )
      raise ValueError(msg)

    see = irt.see(theta, administered_items_array)
    return see < self._min_error


class ItemBankLengthStopper(Stopper):
  """Stopping criterion based on exhausting the entire item bank.

  The test stops when all items in the item bank have been administered to the
  examinee. This is useful for ensuring that an examinee never runs out of items,
  or for creating tests that must use all available items.

  This stopper is typically used in combination with other stoppers (via
  CombinationStopper with OR strategy) to prevent item exhaustion errors.

  Examples
  --------
  >>> # Stop when item bank is exhausted (safety mechanism)
  >>> stopper = ItemBankLengthStopper()

  >>> # Combine with other criteria: stop at 30 items OR when bank exhausted
  >>> from catsim.stopping import CombinationStopper, CombinationStrategy
  >>> stopper = CombinationStopper(
  ...     [MaxItemStopper(30), ItemBankLengthStopper()],
  ...     strategy=CombinationStrategy.OR
  ... )
  """

  def __str__(self) -> str:
    """Get a string representation of the Stopper."""
    return "Item Bank Length Stopper"

  def stop(
    self,
    index: int | None = None,
    _item_bank: ItemBank | None = None,
    administered_items: npt.NDArray[numpy.floating[Any]] | None = None,
    **kwargs: Any,  # noqa: ARG002
  ) -> bool:
    """Check whether all items in the item bank have been administered.

    Parameters
    ----------
    index : int or None, optional
        The index of the current examinee. Default is None.
    _item_bank : ItemBank or None, optional
        The item bank being used. Default is None.
    administered_items : npt.NDArray[numpy.floating[Any]] or None, optional
        A matrix containing the parameters of items that were already administered.
        Default is None.
    **kwargs : dict
        Additional keyword arguments. Not used by this method.

    Returns
    -------
    bool
        True if all items in the item bank have been administered, False otherwise.

    Raises
    ------
    ValueError
        If required parameters are missing.
    """
    # Get the item bank and administered items
    if _item_bank is not None and administered_items is not None:
      bank_size = len(_item_bank)
      n_administered = len(administered_items)
    elif index is not None and self.simulator is not None:
      bank_size = len(self.simulator.item_bank)
      n_administered = len(self.simulator.administered_items[index])
    else:
      msg = (
        "Required parameters are missing. Either _item_bank and administered_items or index and simulator must be "
        "provided."
      )
      raise ValueError(msg)

    # Stop if all items have been administered
    return n_administered >= bank_size


class CombinationStopper(Stopper):
  """Stopping criterion that combines multiple stoppers using AND or OR logic.

  This stopper allows you to combine multiple stopping criteria in a flexible way.
  With OR logic (default), the test stops when ANY of the stoppers indicates stopping.
  With AND logic, the test stops only when ALL stoppers indicate stopping.

  Parameters
  ----------
  stoppers : list[Stopper]
      A list of Stopper instances to combine. Must contain at least one stopper.
  strategy : CombinationStrategy, optional
      The combination strategy to use. Default is CombinationStrategy.OR.

  Examples
  --------
  >>> # Stop when either 30 items OR error < 0.4 (whichever comes first)
  >>> stopper = CombinationStopper(
  ...     [MaxItemStopper(30), MinErrorStopper(0.4)],
  ...     strategy=CombinationStrategy.OR
  ... )

  >>> # Stop only when BOTH 20 items AND error < 0.3 are achieved
  >>> stopper = CombinationStopper(
  ...     [MaxItemStopper(20), MinErrorStopper(0.3)],
  ...     strategy=CombinationStrategy.AND
  ... )

  >>> # Combine three stoppers: stop when any condition is met
  >>> stopper = CombinationStopper(
  ...     [
  ...         MaxItemStopper(50),
  ...         MinErrorStopper(0.3),
  ...         ConfidenceIntervalStopper([-1.0, 0.0, 1.0], confidence=0.90)
  ...     ],
  ...     strategy=CombinationStrategy.OR
  ... )
  """

  def __str__(self) -> str:
    """Get a string representation of the Stopper."""
    strategy_name = "OR" if self._strategy == CombinationStrategy.OR else "AND"
    stopper_names = [str(s) for s in self._stoppers]
    return f"Combination Stopper ({strategy_name}: {', '.join(stopper_names)})"

  def __init__(self, stoppers: list[Stopper], strategy: CombinationStrategy = CombinationStrategy.OR) -> None:
    """Initialize a CombinationStopper.

    Parameters
    ----------
    stoppers : list[Stopper]
        List of Stopper instances to combine. Must contain at least one stopper.
    strategy : CombinationStrategy, optional
        The combination strategy (OR or AND). Default is CombinationStrategy.OR.

    Raises
    ------
    ValueError
        If stoppers list is empty or contains non-Stopper objects.
    TypeError
        If strategy is not a CombinationStrategy enum value.
    """
    if not stoppers:
      msg = "stoppers list must contain at least one Stopper"
      raise ValueError(msg)

    if not all(isinstance(s, Stopper) for s in stoppers):
      msg = "All elements in stoppers list must be Stopper instances"
      raise ValueError(msg)

    if not isinstance(strategy, CombinationStrategy):
      msg = f"strategy must be a CombinationStrategy enum value, got {type(strategy)}"
      raise TypeError(msg)

    super().__init__()
    self._stoppers = list(stoppers)
    self._strategy = strategy

  def stop(
    self,
    index: int | None = None,
    _item_bank: ItemBank | None = None,
    administered_items: npt.NDArray[numpy.floating[Any]] | None = None,
    theta: float | None = None,
    **kwargs: Any,
  ) -> bool:
    """Check whether the combined stopping criterion is met.

    Parameters
    ----------
    index : int or None, optional
        The index of the current examinee. Default is None.
    administered_items : npt.NDArray[numpy.floating[Any]] or None, optional
        A matrix containing the parameters of items that were already administered.
        Default is None.
    theta : float or None, optional
        An ability value. Default is None.
    **kwargs : dict
        Additional keyword arguments passed to individual stoppers.

    Returns
    -------
    bool
        - If strategy is OR: True if ANY stopper indicates stopping
        - If strategy is AND: True if ALL stoppers indicate stopping

    Notes
    -----
    The CombinationStopper extracts data from the simulator (if needed) and calls
    all child stoppers in standalone mode with explicit parameters, so the child
    stoppers don't need to depend on having a simulator reference.
    """
    # If administered_items or theta not provided, get them from simulator
    if administered_items is None and index is not None and self.simulator is not None:
      administered_items = self.simulator.item_bank.get_items(indices=self.simulator.administered_items[index])
      theta = self.simulator.latest_estimations[index]
      # Also get item_bank if not provided
      if _item_bank is None:
        _item_bank = self.simulator.item_bank
    elif administered_items is None or theta is None:
      msg = (
        "Required parameters are missing. Either index and simulator or administered_items and theta must be provided."
      )
      raise ValueError(msg)

    # Call all stoppers with explicit parameters (standalone mode)
    results = []
    for stopper in self._stoppers:
      result = stopper.stop(
        index=None, _item_bank=_item_bank, administered_items=administered_items, theta=theta, **kwargs
      )
      results.append(result)

    # Combine results based on strategy
    if self._strategy == CombinationStrategy.OR:
      return any(results)
    # CombinationStrategy.AND
    return all(results)

  @property
  def stoppers(self) -> list[Stopper]:
    """Get the list of stoppers being combined.

    Returns
    -------
    list[Stopper]
        The list of Stopper instances.
    """
    return self._stoppers

  @property
  def strategy(self) -> CombinationStrategy:
    """Get the combination strategy.

    Returns
    -------
    CombinationStrategy
        The combination strategy (OR or AND).
    """
    return self._strategy


class ConfidenceIntervalStopper(Stopper):
  r"""Stopping criterion based on confidence interval falling within discrete ability intervals.

  This stopper is designed for tests with discrete performance levels (e.g., letter grades
  A, B, C, D, F) defined by intervals on the ability scale. The test stops when the
  confidence interval for the examinee's ability estimate falls entirely within one of
  these discrete intervals, indicating sufficient precision to classify the examinee.

  The figure below illustrates the concept with an example where an examinee's ability
  is estimated at θ = 1.0 with a confidence interval that falls entirely within the
  grade B interval [0.5, 1.5), causing the test to stop.

  .. plot::
      :caption: Confidence interval-based stopping. The shaded area shows the 90% confidence
                interval around the estimated ability (θ = 1.0). Since the entire confidence
                interval falls within the grade B boundaries, the test stops.

      import matplotlib.pyplot as plt
      import numpy as np
      from scipy import stats

      # Setup
      fig, ax = plt.subplots(figsize=(12, 6))

      # Define grade boundaries
      boundaries = [-1.5, -0.5, 0.5, 1.5]
      grade_names = ['F', 'D', 'C', 'B', 'A']
      grade_colors = ['#ff6b6b', '#ffa07a', '#ffd93d', '#95e1d3', '#6bcf7f']

      # Define the ability scale range
      theta_range = np.linspace(-3, 3, 1000)

      # Draw grade regions with shading
      regions = [
        (-3, boundaries[0])
      ] + [
        (boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)
      ] + [
        (boundaries[-1], 3)
      ]

      for i, (start, end) in enumerate(regions):
          ax.axvspan(start, end, alpha=0.15, color=grade_colors[i], label=f'Grade {grade_names[i]}')
          # Add grade label in the middle of each region
          mid = (start + end) / 2
          ax.text(
              mid, 0.85, grade_names[i], ha='center', va='center',
              fontsize=16, fontweight='bold', color=grade_colors[i],
              bbox=dict(
                  boxstyle='round,pad=0.5',
                  facecolor='white',
                  edgecolor=grade_colors[i],
                  linewidth=2
              )
          )

      # Draw boundary lines
      for boundary in boundaries:
          ax.axvline(boundary, color='black', linestyle='--', linewidth=2, alpha=0.5)

      # Example: examinee with theta = 1.0 and SEE = 0.2
      # This gives a CI that falls entirely within grade B [0.5, 1.5)
      estimated_theta = 1.0
      see = 0.2
      confidence = 0.90

      # Calculate confidence interval
      z_score = stats.norm.ppf((1 + confidence) / 2)
      ci_lower = estimated_theta - z_score * see
      ci_upper = estimated_theta + z_score * see

      # Draw normal distribution curve
      normal_curve = stats.norm.pdf(theta_range, estimated_theta, see)
      # Scale it for visibility
      normal_curve_scaled = normal_curve * 0.6 / np.max(normal_curve)

      ax.plot(theta_range, normal_curve_scaled, 'b-', linewidth=2.5, label='Ability distribution')

      # Shade the confidence interval
      mask = (theta_range >= ci_lower) & (theta_range <= ci_upper)
      ax.fill_between(
          theta_range[mask], 0, normal_curve_scaled[mask],
          alpha=0.4, color='blue', label=f'{int(confidence*100)}% Confidence Interval'
      )

      # Mark the estimated theta
      ax.plot(estimated_theta, 0, 'ro', markersize=15, label=f'Estimated θ = {estimated_theta}', zorder=5)
      ax.axvline(estimated_theta, color='red', linestyle='-', linewidth=2, alpha=0.7, ymax=0.65)

      # Mark confidence interval bounds
      ax.axvline(ci_lower, color='blue', linestyle=':', linewidth=2, alpha=0.8)
      ax.axvline(ci_upper, color='blue', linestyle=':', linewidth=2, alpha=0.8)

      # Add arrows and labels for CI bounds
      ax.annotate(
          f'CI Lower\\n{ci_lower:.2f}', xy=(ci_lower, 0.35), xytext=(ci_lower-0.4, 0.45),
          fontsize=10, ha='center', color='blue',
          arrowprops=dict(arrowstyle='->', color='blue', lw=1.5)
      )
      ax.annotate(
          f'CI Upper\\n{ci_upper:.2f}', xy=(ci_upper, 0.35), xytext=(ci_upper+0.4, 0.45),
          fontsize=10, ha='center', color='blue',
          arrowprops=dict(arrowstyle='->', color='blue', lw=1.5)
      )

      # Add decision box
      ax.text(
          1.0, 0.72, 'CI entirely in Grade B\\n→ STOP TEST',
          ha='center', va='center', fontsize=12, fontweight='bold',
          bbox=dict(
              boxstyle='round,pad=0.8',
              facecolor='lightgreen',
              edgecolor='darkgreen',
              linewidth=3
          )
      )

      # Formatting
      ax.set_xlabel('Ability (θ)', fontsize=14, fontweight='bold')
      ax.set_ylabel('Probability Density', fontsize=14, fontweight='bold')
      ax.set_title('Confidence Interval-Based Stopping Criterion', fontsize=16, fontweight='bold', pad=20)
      ax.set_xlim(-3, 3)
      ax.set_ylim(0, 0.95)
      ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
      ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

      plt.tight_layout()
      plt.show()

  Parameters
  ----------
  interval_bounds : list[float]
      Sorted list of boundary points defining the discrete intervals on the ability scale.
      For example, [-2.0, -0.5, 0.5, 2.0] defines 5 intervals:
      (-inf, -2.0), [-2.0, -0.5), [-0.5, 0.5), [0.5, 2.0), [2.0, inf)
  confidence : float, optional
      The confidence level for computing the confidence interval, must be between 0 and 1.
      Default is 0.95 (95% confidence).

  Examples
  --------
  >>> # Define grade boundaries: F (<-1), D [-1, 0), C [0, 1), B [1, 2), A (>=2)
  >>> stopper = ConfidenceIntervalStopper([-1.0, 0.0, 1.0, 2.0], confidence=0.95)
  >>> # Test stops when 95% CI is entirely within one grade interval
  """

  def __str__(self) -> str:
    """Get a string representation of the Stopper."""
    return "Confidence Interval Stopper"

  def __init__(self, interval_bounds: list[float], confidence: float = 0.95) -> None:
    """Initialize a ConfidenceIntervalStopper.

    Parameters
    ----------
    interval_bounds : list[float]
        Sorted list of boundary points on the ability scale. Must contain at least one value.
    confidence : float, optional
        Confidence level for the interval computation (between 0 and 1). Default is 0.95.

    Raises
    ------
    ValueError
        If interval_bounds is empty, not sorted, or confidence is not between 0 and 1.
    """
    super().__init__()

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

    self._interval_bounds = list(interval_bounds)
    self._confidence = confidence

  def stop(
    self,
    index: int | None = None,
    _item_bank: ItemBank | None = None,
    administered_items: npt.NDArray[numpy.floating[Any]] | None = None,
    theta: float | None = None,
    **kwargs: Any,  # noqa: ARG002
  ) -> bool:
    """Check whether the confidence interval falls entirely within a discrete interval.

    Parameters
    ----------
    index : int or None, optional
        The index of the current examinee. Default is None.
    administered_items : npt.NDArray[numpy.floating[Any]] or None, optional
        A matrix containing the parameters of items that were already administered.
        Default is None.
    theta : float or None, optional
        An ability value to which the confidence interval will be computed. Default is None.
    **kwargs : dict
        Additional keyword arguments. Not used by this method.

    Returns
    -------
    bool
        True if the confidence interval for the ability estimate falls entirely within
        one of the discrete intervals, False otherwise.

    Raises
    ------
    ValueError
        If required parameters are missing.
    """
    # Get administered items and theta
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

    # Need at least one item to compute confidence interval
    if len(administered_items_array) == 0:
      return False

    # Compute confidence interval
    lower_bound, upper_bound = irt.confidence_interval(theta, administered_items_array, self._confidence)

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
