"""Module containing functions relevant to the process of simulating the application of adaptive tests.

Most of this module is based on the work of [Bar10]_.
"""

import time
from typing import Any

import numpy
import numpy.typing as npt
from tqdm import tqdm

from . import cat, irt
from .estimation import BaseEstimator
from .initialization import BaseInitializer
from .item_bank import ItemBank
from .selection import BaseSelector, FiniteSelector
from .stopping import BaseStopper

# Re-export component base classes for backward compatibility
Estimator = BaseEstimator
Selector = BaseSelector


class Simulator:
  r"""Class representing the CAT simulator.

  The Simulator gathers several objects that describe the full simulation process
  (initializer, selector, estimator, stopper) and simulates one or more computerized
  adaptive tests.

  Parameters
  ----------
  item_bank : ItemBank or numpy.ndarray
      An ItemBank object containing item parameters. If a numpy.ndarray is provided,
      it will be automatically converted to an ItemBank.
  examinees : int or npt.ArrayLike
      Either an integer with the number of examinees (whose real :math:`\theta` values
      will be sampled from a normal distribution), or an array-like (list, tuple, or
      numpy array) containing the examinees' true :math:`\theta` values (float type).
  initializer : BaseInitializer or None, optional
      BaseInitializer to use during the simulation. Default is None.
  selector : Selector or None, optional
      Selector to use during the simulation. Default is None.
  estimator : Estimator or None, optional
      Estimator to use during the simulation. Default is None.
  stopper : Stopper or None, optional
      Stopper to use during the simulation. Default is None.
  seed : int, optional
      Seed used by the numpy random number generator during the simulation procedure.
      Default is 0.
  """

  def __init__(
    self,
    item_bank: ItemBank | npt.NDArray[numpy.floating[Any]],
    examinees: int | npt.ArrayLike,
    initializer: BaseInitializer | None = None,
    selector: BaseSelector | None = None,
    estimator: BaseEstimator | None = None,
    stopper: BaseStopper | None = None,
    seed: int = 0,
  ) -> None:
    """Initialize a Simulator object.

    Parameters
    ----------
    item_bank : ItemBank or numpy.ndarray
        ItemBank object or item parameter matrix. If a numpy array is provided,
        it will be converted to an ItemBank automatically.
    examinees : int or npt.ArrayLike
        Either an integer with number of examinees or array-like (list, tuple, or
        numpy array) of examinee abilities (float type).
    initializer : Initializer or None, optional
        Initializer to use during the simulation. Default is None.
    selector : Selector or None, optional
        Selector to use during the simulation. Default is None.
    estimator : Estimator or None, optional
        Estimator to use during the simulation. Default is None.
    stopper : Stopper or None, optional
        Stopper to use during the simulation. Default is None.
    seed : int, optional
        Seed used by the numpy random number generator during the simulation procedure.
        Default is 0.
    """
    # Convert numpy array to ItemBank if necessary
    if isinstance(item_bank, numpy.ndarray):
      item_bank = ItemBank(item_bank)
    elif not isinstance(item_bank, ItemBank):
      msg = "item_bank must be an ItemBank or numpy.ndarray"
      raise TypeError(msg)

    self._duration = 0.0
    self._item_bank = item_bank
    # Track item usage counts for efficient exposure rate updates
    self._item_usage_counts = numpy.zeros(item_bank.n_items, dtype=int)

    self._bias = 0.0
    self._mse = 0.0
    self._rmse = 0.0
    self._overlap_rate = 0.0

    self._initializer = initializer
    self._selector = selector
    self._estimator = estimator
    self._stopper = stopper

    self.__rng = numpy.random.default_rng(seed=seed)

    # `examinees` is passed to its special setter
    self._examinees = self._to_distribution(examinees)

    self._estimations: list[list[float]] = [[] for _ in range(self.examinees.shape[0])]
    self._administered_items: list[list[int]] = [[] for _ in range(self.examinees.shape[0])]
    self._response_vectors: list[list[bool]] = [[] for _ in range(self.examinees.shape[0])]

  @property
  def item_bank(self) -> ItemBank:
    """ItemBank used by the simulator.

    Returns
    -------
    ItemBank
        The ItemBank containing all item parameters and exposure rates.
    """
    return self._item_bank

  @property
  def items(self) -> npt.NDArray[numpy.floating[Any]]:
    """Item matrix used by the simulator (for backward compatibility).

    Returns
    -------
    npt.NDArray[numpy.floating[Any]]
        The underlying item parameter matrix from the ItemBank.
    """
    return self._item_bank.items

  @property
  def administered_items(self) -> list[list[int]]:
    """A list of lists with the indexes of items administered to each examinee during the simulation."""
    return self._administered_items

  @property
  def estimations(self) -> list[list[float]]:
    r"""A list of lists with all estimated :math:`\hat\theta` values for all examinees during each step of the test."""
    return self._estimations

  @property
  def response_vectors(self) -> list[list[bool]]:
    """List of boolean lists containing the examinees answers to all items."""
    return self._response_vectors

  @property
  def latest_estimations(self) -> list[float]:
    r"""Final estimated :math:`\hat\theta` values for all examinees."""
    # Filter out any None values - all examinees should have at least one estimation
    result = []
    for ests in self._estimations:
      if len(ests) > 0:
        result.append(ests[-1])
      else:
        # Fallback: if no estimations, use 0.0 (should not happen in normal operation)
        result.append(0.0)
    return result

  @property
  def duration(self) -> float:
    """Duration of the simulation, in seconds."""
    return self._duration

  @property
  def overlap_rate(self) -> float:
    """Overlap rate of the test, if it is of finite length."""
    return self._overlap_rate

  @property
  def initializer(self) -> BaseInitializer | None:
    """Get the initializer used during the simulation.

    Returns
    -------
    BaseInitializer or None
        The initializer used during the simulation.
    """
    return self._initializer

  @property
  def selector(self) -> BaseSelector | None:
    """Get the selector used during the simulation.

    Returns
    -------
    Selector or None
        The selector used during the simulation.
    """
    return self._selector

  @property
  def estimator(self) -> BaseEstimator | None:
    """Get the estimator used during the simulation.

    Returns
    -------
    Estimator or None
        The estimator used during the simulation.
    """
    return self._estimator

  @property
  def stopper(self) -> BaseStopper | None:
    """Get the stopper used during the simulation.

    Returns
    -------
    Stopper or None
        The stopper used during the simulation.
    """
    return self._stopper

  @property
  def bias(self) -> float:
    """Get the bias between the estimated and true abilities.

    This property is only available after :py:func:`simulate` has been successfully called. For more information on
    estimation bias, see :py:func:`catsim.cat.bias`
    """
    return self._bias

  @property
  def mse(self) -> float:
    """Get the mean-squared error between the estimated and true abilities.

    This property is only available after :py:func:`simulate` has been successfully called. For more information on the
    mean-squared error of estimation, see :py:func:`catsim.cat.mse`.
    """
    return self._mse

  @property
  def rmse(self) -> float:
    """Get the root mean-squared error between the estimated and true abilities.

    This property is only available after :py:func:`simulate` has been successfully called. For more information on the
    root mean-squared error of estimation, see :py:func:`catsim.cat.rmse`.
    """
    return self._rmse

  @property
  def examinees(self) -> npt.NDArray[numpy.floating[Any]]:
    r""":py:type:numpy.ndarray containing examinees true ability values (:math:`\theta`)."""
    return self._examinees

  @property
  def rng(self) -> numpy.random.Generator:
    """Get the random number generator used by the simulator."""
    return self.__rng

  @examinees.setter
  def examinees(self, x: int | npt.ArrayLike) -> None:
    self._examinees = self._to_distribution(x)

  def _to_distribution(self, x: int | npt.ArrayLike) -> npt.NDArray[numpy.floating[Any]]:
    """Generate examinees from a distribution, if the Simulator was initialized with an int.

    Parameters
    ----------
    x : int or npt.ArrayLike
        Variable representing the number of examinees (int) or the actual ability values
        (array-like: list, tuple, or numpy array of float type).

    Returns
    -------
    numpy.ndarray
        Examinees as a numpy array.

    Raises
    ------
    TypeError
        If the examinees are not an int, list of floats, or one-dimensional numpy array.
    ValueError
        If the number of examinees is invalid (non-positive int or empty list/array).
    """
    if isinstance(x, int):
      if x <= 0:
        msg = f"Number of examinees must be positive, got {x}"
        raise ValueError(msg)
      if self._item_bank is not None:
        mean = numpy.mean(self._item_bank.difficulty)
        stddev = numpy.std(self._item_bank.difficulty)
        dist = self.__rng.normal(mean, stddev, x)
      else:
        dist = self.__rng.normal(0, 1, x)
    else:
      # Convert array-like to numpy array
      x_array = numpy.asarray(x)
      if x_array.ndim != 1:
        msg = "Examinees array must be one-dimensional"
        raise TypeError(msg)
      if x_array.size == 0:
        msg = "Array of examinees cannot be empty"
        raise ValueError(msg)
      dist = x_array

    return dist

  def simulate(
    self,
    initializer: BaseInitializer | None = None,
    selector: BaseSelector | None = None,
    estimator: BaseEstimator | None = None,
    stopper: BaseStopper | None = None,
    verbose: bool = False,
  ) -> None:
    r"""Simulate a computerized adaptive testing application to one or more examinees.

    The simulation process iterates through each examinee, initializing their ability,
    selecting items, recording responses, estimating abilities, and stopping when the
    criterion is met.

    Parameters
    ----------
    initializer : Initializer or None, optional
        An initializer that selects examinees' initial :math:`\theta_0`. Default is None.
    selector : Selector or None, optional
        A selector that selects new items to be presented to examinees. Default is None.
    estimator : Estimator or None, optional
        An estimator that reestimates examinees abilities after each item is applied.
        Default is None.
    stopper : Stopper or None, optional
        An object with a stopping criterion for the test. Default is None.
    verbose : bool, optional
        Whether to periodically print a message regarding the progress of the simulation.
        Good for longer simulations. Default is False.

    Raises
    ------
    ValueError
        If any of initializer, selector, estimator, or stopper is None (either passed
        or from constructor).

    Examples
    --------
    >>> from catsim.initialization import RandomInitializer
    >>> from catsim.selection import MaxInfoSelector
    >>> from catsim.estimation import NumericalSearchEstimator
    >>> from catsim.stopping import MinErrorStopper
    >>> from catsim.simulation import Simulator
    >>> from catsim.item_bank import ItemBank
    >>> initializer = RandomInitializer()
    >>> selector = MaxInfoSelector()
    >>> estimator = NumericalSearchEstimator()
    >>> stopper = MinErrorStopper(0.4, max_items=20)
    >>> Simulator(ItemBank.generate_item_bank(100), 10).simulate(initializer, selector, estimator, stopper)
    """
    if initializer is not None:
      self._initializer = initializer
    if selector is not None:
      self._selector = selector
    if estimator is not None:
      self._estimator = estimator
    if stopper is not None:
      self._stopper = stopper

    if self._initializer is None:
      msg = "Initializer is required for simulation"
      raise ValueError(msg)
    if self._selector is None:
      msg = "Selector is required for simulation"
      raise ValueError(msg)
    if self._estimator is None:
      msg = "Estimator is required for simulation"
      raise ValueError(msg)
    if self._stopper is None:
      msg = "Stopper is required for simulation"
      raise ValueError(msg)

    for s in [self._initializer, self._selector, self._estimator, self._stopper]:
      s.simulator = self

    # Reset the item bank to clear any previous simulation data
    self._item_bank.reset()

    pbar = None
    if verbose:
      print(
        f"Starting simulation: {self._initializer} {self._selector} "
        f"{self._estimator} {self._stopper} {self._item_bank.n_items} items"
      )
      pbar = tqdm(total=len(self.examinees))

    start_time = time.time()

    for current_examinee, true_theta in enumerate(self.examinees):
      if verbose and pbar is not None:
        pbar.update()

      est_theta = self._initializer.initialize(index=current_examinee)
      self._estimations[current_examinee].append(est_theta)

      while not self._stopper.stop(index=current_examinee):
        selected_item = self._selector.select(index=current_examinee)

        # if the selector returns None, it means the selector and not the stopper, is asking the test to stop
        # this happens e.g. if the item bank or the available strata end before the minimum error is achieved
        if selected_item is None:
          break

        # validate selected item index
        is_valid_type = isinstance(selected_item, (int, numpy.integer))
        is_valid_range = 0 <= selected_item < self._item_bank.n_items
        if not (is_valid_type and is_valid_range):
          msg = (
            f"Invalid item index {selected_item} returned by selector. "
            f"Must be between 0 and {self._item_bank.n_items - 1}"
          )
          raise ValueError(msg)

        # simulates the examinee's response via the four-parameter
        # logistic function
        item_params = self._item_bank.get_item(selected_item)
        response = (
          irt.icc(
            true_theta,
            item_params[0],  # a
            item_params[1],  # b
            item_params[2],  # c
            item_params[3],  # d
          )
          >= self.__rng.uniform()
        )

        self._response_vectors[current_examinee].append(response)

        # adds the item selected by the selector to the pool of administered items
        self._administered_items[current_examinee].append(selected_item)

        # estimate the new theta using the given estimator
        est_theta = self._estimator.estimate(index=current_examinee)

        # update item usage count and exposure rate efficiently
        self._item_usage_counts[selected_item] += 1
        self._item_bank.update_exposure_rate(
          selected_item, self._item_usage_counts[selected_item] / len(self.examinees)
        )

        self._estimations[current_examinee].append(est_theta)

    self._duration = time.time() - start_time

    if verbose and pbar is not None:
      pbar.close()
      print(f"Simulation took {self._duration} seconds")

    self._bias = cat.bias(self.examinees, self.latest_estimations)
    self._mse = cat.mse(self.examinees, self.latest_estimations)
    self._rmse = cat.rmse(self.examinees, self.latest_estimations)

    # overlap is computed only if all examinees answered the same amount of items
    # maybe there is a way to calculate it with tests of different lengths,
    # but I did not find it in the literature
    test_size = None
    len_first = len(self._administered_items[0]) if self._administered_items else None
    if isinstance(selector, FiniteSelector):
      test_size = selector.test_size
    elif all(len(i) == len_first for i in self._administered_items):
      test_size = len_first
    if test_size is not None:
      self._overlap_rate = cat.overlap_rate(self._item_bank.exposure_rates, test_size)


if __name__ == "__main__":
  import doctest

  doctest.testmod()
