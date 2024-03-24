"""Module containing functions relevant to the process of simulating the application of adaptive tests.

Most of this module is based on the work of [Bar10]_.
"""

import time
from abc import ABC, abstractmethod
from typing import Any

import numpy
from tqdm import tqdm

from . import cat, irt


class Simulable:
  """Base class representing one of the Simulator components that will receive a reference back to it."""

  def __init__(self) -> None:
    """Initialize a Simulable object."""
    super(Simulable).__init__()
    self._simulator: Simulator = None

  @property
  def simulator(self) -> "Simulator":
    """Set or get the simulator object.

    :raises TypeError: If the simulator is not of type catsim.simulation.Simulator.
    :return: The :py:class:`Simulator` instance tied to this Simulable.
    :rtype: Simulator
    """
    if self._simulator is not None and not isinstance(self._simulator, Simulator):
      msg = "simulator has to be of type catsim.simulation.Simulator"
      raise TypeError(msg)
    return self._simulator

  @simulator.setter
  def simulator(self, x: "Simulator") -> None:
    if not isinstance(x, Simulator):
      msg = "simulator has to be of type catsim.simulation.Simulator"
      raise TypeError(msg)
    self._simulator = x
    self.preprocess()

  def preprocess(self) -> None:
    """Override this method to initialize any values the `Simulable` might use for the duration of the simulation.

    `preprocess` is called after a value is set for the `simulator` property. If a new value if
    attributed to `simulator`, this method is called again, guaranteeing that internal properties of the
    `Simulable` are re-initialized as necessary.
    """

  def _prepare_args(
    self,
    return_items: bool = False,
    return_response_vector: bool = False,
    return_est_theta: bool = False,
    **kwargs: dict[str, Any],
  ) -> tuple:
    """Prepare input arguments for all Simulable objects.

    This function takes in several optional parameters and returns a tuple based on the conditions met.

    Parameters:
    :param return_items: bool, whether to return items
    :param return_response_vector: bool, whether to return response vector
    :param return_est_theta: bool, whether to return estimated theta
    :param **kwargs: dict[str, Any], additional keyword arguments
    :returns: A tuple containing the specified results based on the conditions.
    """
    using_simulator_props = kwargs.get("index") is not None and self.simulator is not None
    if not using_simulator_props and (
      kwargs.get("items") is None
      or kwargs.get("administered_items") is None
      or (return_est_theta and kwargs.get("est_theta") is None)
    ):
      msg = (
        "Either pass an index for the simulator or all of the other optional parameters to use this component "
        "independently."
      )
      raise ValueError(msg)

    result = []
    if using_simulator_props:
      index = kwargs["index"]
      if return_items:
        result.append(self.simulator.items)
      result.append(self.simulator.administered_items[index])
      if return_response_vector:
        result.append(self.simulator.response_vectors[index])
      if return_est_theta:
        result.append(self.simulator.latest_estimations[index])
    else:
      if return_items:
        result.append(kwargs["items"])
      result.append(kwargs["administered_items"])
      if return_response_vector:
        result.append(kwargs["response_vector"])
      if return_est_theta:
        result.append(kwargs["est_theta"])
    return tuple(result)


class Initializer(Simulable, ABC):
  """Base class for CAT initializers."""

  def __init__(self) -> None:
    """Initialize an Initializer object."""
    super().__init__()

  @abstractmethod
  def initialize(self, index: int) -> float:
    r"""Select an examinee's initial :math:`\theta` value.

    :param index: the index of the current examinee
    :returns: examinee's initial :math:`\theta` value
    """


class Selector(Simulable, ABC):
  """Base class representing a CAT item selector."""

  def __init__(self) -> None:
    """Initialize a Selector object."""
    super().__init__()

  @staticmethod
  def _get_non_administered(item_indices: list[int], administered_item_indices: list[int]) -> list:
    """Get a list of items that were not administered from a list of indices.

    :param item_indices: a list of integers, corresponding to item indices
    :type item_indices: List[int]
    :param administered_item_indices: a list of integers, corresponding to the indices of items that were alredy
    administered to a given examinee
    :type administered_item_indices: List[int]
    :return: a list of items, corresponding to the indices that are in `item_indices` but not in
    `administered_item_indices`, in the same order they were passed in `item_indices`
    :rtype: List[int]
    """
    return [x for x in item_indices if x not in administered_item_indices]

  @staticmethod
  def _sort_by_info(items: numpy.ndarray, est_theta: float) -> list:
    """Sort items by their information value, given a ability value.

    :param items: an item parameter matrix
    :type items: numpy.ndarray
    :param est_theta: an examinee's ability
    :type est_theta: float
    :return: List[int] containing the indices of items, sorted in descending order by their information values (much
    like the return of `numpy.argsort`)
    :rtype: List[int]
    """
    if irt.detect_model(items) == 1:
      # when the logistic model has the number of parameters <= 2,
      # all items have highest information where theta = b
      ordered_items = Selector._sort_by_b(items, est_theta)
    else:
      # else, sort item indexes by their information value descending and remove indexes of administered items
      ordered_items = list((-irt.inf_hpc(est_theta, items)).argsort())
    return ordered_items

  @staticmethod
  def _sort_by_b(items: numpy.ndarray, est_theta: float) -> list:
    """Sort items by how close their difficulty parameter is in relaiton to an examinee's ability.

    :param items: an item parameter matrix
    :type items: numpy.ndarray
    :param est_theta: an examinee's ability
    :type est_theta: float
    :return: list containing the indices of items, sorted by how close their difficulty parameter is in relation to
    :param:`est_theta` (much like the return of `numpy.argsort`)
    :rtype: List[int]
    """
    return list(numpy.abs(items[:, 1] - est_theta).argsort())

  @abstractmethod
  def select(self, index: int | None = None) -> int | None:
    """Returns the index of the next item to be administered.

    :param index: the index of the current examinee in the simulator.
    :returns: index of the next item to be applied or `None` if there are no more items to be presented.
    """


class FiniteSelector(Selector, ABC):
  """Base class representing a CAT item selector."""

  def __init__(self, test_size: int) -> None:
    """Initialize a FiniteSelector object.

    :param test_size: Number of items to be administered in the test.
    :type test_size: int
    """
    self._test_size = test_size
    self._overlap_rate = None
    super().__init__()

  @property
  def test_size(self) -> int:
    """Get the number of items to be administered in the test.

    :return: Number of items to be administered in the test.
    :rtype: int
    """
    return self._test_size

  @property
  def overlap_rate(self) -> float:
    """Get the overlap rate of the test, if it is of finite length.

    :return: Overlap rate of the test.
    :rtype: float
    """
    return self._overlap_rate


class Estimator(Simulable, ABC):
  """Base class for ability estimators."""

  def __init__(self, verbose: bool = False) -> None:
    """Initialize an Estimator object.

    :param verbose: Whether to be verbose during execution, defaults to False
    :type verbose: bool, optional
    """
    super().__init__()
    self._calls = 0
    self._evaluations = 0
    self._verbose = verbose

  @abstractmethod
  def estimate(self, index: int) -> float:
    r"""Compute the theta value that maximizes the log-likelihood function for the given examinee in a test.

    :param index: index of the current examinee in the simulator
    :returns: the current :math:`\hat\theta`
    """

  @property
  def calls(self) -> int:
    """Get how many times the estimator has been called to maximize/minimize the log-likelihood function.

    :returns: number of times the estimator has been called to maximize/minimize the log-likelihood function
    """
    return self._calls

  @property
  def evaluations(self) -> int:
    """Get the total number of times the estimator has evaluated the log-likelihood function during its existence.

    :returns: number of function evaluations
    """
    return self._evaluations

  @property
  def avg_evaluations(self) -> float:
    """Get the average number of function evaluations for all tests the estimator has been used.

    :returns: average number of function evaluations
    """
    return self._evaluations / self._calls


class Stopper(Simulable, ABC):
  """Base class for CAT stop criterion."""

  def __init__(self) -> None:
    """Initialize a Stopper object."""
    super().__init__()

  @abstractmethod
  def stop(self, index: int) -> bool:
    """Checks whether the test reached its stopping criterion for the given user.

    :param index: the index of the current examinee
    :returns: `True` if the test met its stopping criterion, else `False`
    """


class Simulator:
  r"""Class representing the simulator.

  It gathers several objects that describe the full simulation process and simulates one or more computerized adaptive
  tests.

  :param items: a matrix containing item parameters
  :param examinees: an integer with the number of examinees, whose real :math:`\theta` values will be
                    sampled from a normal distribution; or a :py:type:list containing said
                    :math:`\theta_0` values
  """

  def __init__(
    self,
    items: numpy.ndarray,
    examinees: int | list[float] | numpy.ndarray,
    initializer: Initializer = None,
    selector: Selector = None,
    estimator: Estimator = None,
    stopper: Stopper = None,
  ) -> None:
    """Initialize a Simulator object.

    :param items: Matrix containing item parameters.
    :type items: numpy.ndarray
    :param examinees: Integer with number of examinees or list of integers with examinee abilities.
    :type examinees: int | list[float] | numpy.ndarray
    :param initializer: Initializer to use during the simulation, defaults to None.
    :type initializer: Initializer, optional
    :param selector: Selector to use during the simulation, defaults to None.
    :type selector: Selector, optional
    :param estimator: Estimator to use during the simulation, defaults to None.
    :type estimator: Estimator, optional
    :param stopper: Stopper to use during the simulation, defaults to None.
    :type stopper: Stopper, optional
    """
    irt.validate_item_bank(items)

    # adds a column for each item's exposure rate
    if items.shape[1] < 5:  # noqa: PLR2004
      items = numpy.append(items, numpy.zeros([items.shape[0], 1]), axis=1)

    self._duration = 0.0
    self._items = items

    self._bias = 0.0
    self._mse = 0.0
    self._rmse = 0.0
    self._overlap_rate = 0.0

    self._initializer = initializer
    self._selector = selector
    self._estimator = estimator
    self._stopper = stopper

    self.__rng = numpy.random.default_rng()

    # `examinees` is passed to its special setter
    self._examinees = self._to_distribution(examinees)

    self._estimations: list[list[int]] = [[] for _ in range(self.examinees.shape[0])]
    self._administered_items: list[list[int]] = [[] for _ in range(self.examinees.shape[0])]
    self._response_vectors: list[list[bool]] = [[] for _ in range(self.examinees.shape[0])]

  @property
  def items(self) -> numpy.ndarray:
    """Item matrix used by the simulator.

    If the simulation already occurred, a column containing item exposure rates will be added to the matrix.
    """
    return self._items

  @property
  def administered_items(self) -> list:
    """A list of lists with the indexes of items administered to each examinee during the simulation."""
    return self._administered_items

  @property
  def estimations(self) -> list:
    r"""A list of lists with all estimated :math:`\hat\theta` values for all examinees during each step of the test."""
    return self._estimations

  @property
  def response_vectors(self) -> list:
    """List of boolean lists containing the examinees answers to all items."""
    return self._response_vectors

  @property
  def latest_estimations(self) -> list:
    r"""Final estimated :math:`\hat\theta` values for all examinees."""
    return [ests[-1] if len(ests) > 0 else None for ests in self._estimations]

  @property
  def duration(self) -> float:
    """Duration of the simulation, in seconds."""
    return self._duration

  @property
  def overlap_rate(self) -> float:
    """Overlap rate of the test, if it is of finite length."""
    return self._overlap_rate

  @property
  def initializer(self) -> Initializer | None:
    """Get the initializer used during the simulation.

    :return: The initializer used during the simulation.
    :rtype: Initializer | None
    """
    return self._initializer

  @property
  def selector(self) -> Selector | None:
    """Get the selector used during the simulation.

    :return: The selector used during the simulation.
    :rtype: Selector | None
    """
    return self._selector

  @property
  def estimator(self) -> Estimator | None:
    """Get the estimator used during the simulation.

    :return: The estimator used during the simulation.
    :rtype: Estimator | None
    """
    return self._estimator

  @property
  def stopper(self) -> Stopper | None:
    """Get the stopper used during the simulation.

    :return: The stopper used during the simulation.
    :rtype: Stopper | None
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
  def examinees(self) -> numpy.ndarray:
    r""":py:type:numpy.ndarray containing examinees true ability values (:math:`\theta`)."""
    return self._examinees

  @examinees.setter
  def examinees(self, x: int | list[float] | numpy.ndarray) -> None:
    self._examinees = self._to_distribution(x)

  def _to_distribution(self, x: int | list[float] | numpy.ndarray) -> numpy.ndarray:
    """Generate examinees from a distribution, if the Simulator was initialized with an int.

    :param x: Variable representing the number of examinees.
    :type x: int | list[float] | numpy.ndarray
    :raises TypeError: If the examinees are not an int, list of floats or one-dimensional numpy array.
    :return: Examinees as a numpy array.
    :rtype: numpy.ndarray
    """
    if isinstance(x, int):
      if self._items is not None:
        mean = numpy.mean(self._items[:, 1])
        stddev = numpy.std(self._items[:, 1])
        dist = self.__rng.normal(mean, stddev, x)
      else:
        dist = self.__rng.normal(0, 1, x)
    elif isinstance(x, list):
      dist = numpy.array(x)
    elif isinstance(x, numpy.ndarray) and x.ndim == 1:
      dist = x
    else:
      msg = "Examinees must be an int, list of floats or one-dimensional numpy array"
      raise TypeError(msg)

    return dist

  def simulate(
    self,
    initializer: Initializer = None,
    selector: Selector = None,
    estimator: Estimator = None,
    stopper: Stopper = None,
    verbose: bool = False,
  ) -> None:
    r"""Simulate a computerized adaptive testing application to one or more examinees.

    :param initializer: an initializer that selects examinees :math:`\theta_0`
    :param selector: a selector that selects new items to be presented to examinees
    :param estimator: an estimator that reestimates examinees abilities after each item is applied
    :param stopper: an object with a stopping criteria for the test
    :param verbose: whether to periodically print a message regarding the progress of the simulation.
                    Good for longer simulations.

    >>> from catsim.initialization import RandomInitializer
    >>> from catsim.selection import MaxInfoSelector
    >>> from catsim.estimation import NumericalSearchEstimator
    >>> from catsim.stopping import MaxItemStopper
    >>> from catsim.simulation import Simulator
    >>> from catsim.cat import generate_item_bank
    >>> initializer = RandomInitializer()
    >>> selector = MaxInfoSelector()
    >>> estimator = NumericalSearchEstimator()
    >>> stopper = MaxItemStopper(20)
    >>> Simulator(generate_item_bank(100), 10).simulate(initializer, selector, estimator, stopper)
    """
    if initializer is not None:
      self._initializer = initializer
    if selector is not None:
      self._selector = selector
    if estimator is not None:
      self._estimator = estimator
    if stopper is not None:
      self._stopper = stopper

    assert self._initializer is not None
    assert self._selector is not None
    assert self._estimator is not None
    assert self._stopper is not None

    for s in [self._initializer, self._selector, self._estimator, self._stopper]:
      s.simulator = self

    if verbose:
      print(
        f"Starting simulation: {self._initializer} {self._selector} "
        f"{self._estimator} {self._stopper} {self._items.shape[0]} items"
      )
      pbar = tqdm(total=len(self.examinees))

    start_time = time.time()

    for current_examinee, true_theta in enumerate(self.examinees):
      if verbose:
        pbar.update()

      est_theta = self._initializer.initialize(current_examinee)
      self._estimations[current_examinee].append(est_theta)

      while not self._stopper.stop(current_examinee):
        selected_item = self._selector.select(current_examinee)

        # if the selector returns None, it means the selector and not the stopper, is asking the test to stop
        # this happens e.g. if the item bank or or the available strata end before the minimum error is achieved
        if selected_item is None:
          break

        # simulates the examinee's response via the four-parameter
        # logistic function
        response = (
          irt.icc(
            true_theta,
            self.items[selected_item][0],
            self.items[selected_item][1],
            self.items[selected_item][2],
            self.items[selected_item][3],
          )
          >= self.__rng.uniform()
        )

        self._response_vectors[current_examinee].append(response)

        # adds the item selected by the selector to the pool of administered items
        self._administered_items[current_examinee].append(selected_item)

        # estimate the new theta using the given estimator
        est_theta = self._estimator.estimate(current_examinee)

        # count occurrences of this item in all tests
        item_occurrences = numpy.sum([
          selected_item in administered_list for administered_list in self._administered_items
        ])

        # update the exposure value for this item
        # r = number of tests item has been used on / total number of tests
        self.items[selected_item, 4] = item_occurrences / len(self.examinees)

        self._estimations[current_examinee].append(est_theta)

    self._duration = time.time() - start_time

    if verbose:
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
      self._overlap_rate = cat.overlap_rate(self.items[:, 4], test_size)


if __name__ == "__main__":
  import doctest

  doctest.testmod()
