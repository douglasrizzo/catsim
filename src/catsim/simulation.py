"""Module containing functions relevant to the process of simulating the application of adaptive tests.

Most of this module is based on the work of [Bar10]_.
"""

import time
from abc import ABC, abstractmethod
from typing import Any

import numpy
import numpy.typing as npt
from tqdm import tqdm

from . import cat, irt
from .item_bank import ItemBank


class Simulable(ABC):  # noqa: B024
  """Base class representing one of the Simulator components that will receive a reference back to it.

  This class provides the infrastructure for components (Initializers, Selectors,
  Estimators, and Stoppers) to access the Simulator instance they belong to.

  Notes
  -----
  This class inherits from ABC to indicate it's meant to be subclassed, though it doesn't
  define abstract methods itself. Concrete abstract methods are defined in its subclasses
  (Initializer, Selector, Estimator, Stopper).
  """

  def __init__(self) -> None:
    """Initialize a Simulable object."""
    super().__init__()
    self._simulator: Simulator | None = None

  @property
  def simulator(self) -> "Simulator":
    """Set or get the simulator object.

    Returns
    -------
    Simulator
        The :py:class:`Simulator` instance tied to this Simulable.

    Raises
    ------
    TypeError
        If the simulator is not of type catsim.simulation.Simulator or is None.
    """
    if self._simulator is None:
      msg = "simulator has not been set"
      raise TypeError(msg)
    if not isinstance(self._simulator, Simulator):
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

  def preprocess(self) -> None:  # noqa: B027
    """Override this method to perform any initialization the `Simulable` might need for the simulation.

    `preprocess` is called after a value is set for the `simulator` property. If a new
    value is attributed to `simulator`, this method is called again, guaranteeing that
    internal properties of the `Simulable` are re-initialized as necessary.

    Notes
    -----
    The default implementation does nothing. Subclasses should override this method
    if they need to perform setup operations that require access to the simulator.
    """

  def _prepare_args(
    self,
    return_item_bank: bool = False,
    return_administered_items: bool = False,
    return_response_vector: bool = False,
    return_est_theta: bool = False,
    return_rng: bool = False,
    **kwargs: Any,
  ) -> tuple:
    """Prepare input arguments for all Simulable objects.

    This helper method extracts required arguments either from the simulator (if running
    within a simulation) or from the provided kwargs (if used standalone).

    Parameters
    ----------
    return_item_bank : bool, optional
        Whether to return the ItemBank. Default is False.
    return_administered_items : bool, optional
        Whether to return the list of administered item indices. Default is False.
    return_response_vector : bool, optional
        Whether to return the response vector. Default is False.
    return_est_theta : bool, optional
        Whether to return the estimated theta value. Default is False.
    return_rng : bool, optional
        Whether to return the random number generator. Default is False.
    **kwargs : dict
        Additional keyword arguments that may contain the required values when not
        using a simulator.

    Returns
    -------
    tuple
        A tuple containing the specified results based on the conditions.

    Raises
    ------
    ValueError
        If required arguments are missing when not using a simulator.
    """
    using_simulator_props = kwargs.get("index") is not None and self.simulator is not None
    result = []
    if not using_simulator_props:
      msg = "No simulator in use, but optional arguments missing: "
      missing_args = []

      for ret, val in (
        (return_item_bank, "item_bank"),
        (return_administered_items, "administered_items"),
        (return_response_vector, "response_vector"),
        (return_est_theta, "est_theta"),
        (return_rng, "rng"),
      ):
        if ret:
          try:
            result.append(kwargs[val])
          except KeyError:
            missing_args.append(val)
      if len(missing_args) > 0:
        msg += ", ".join(missing_args)
        raise ValueError(msg)

    else:
      index = kwargs["index"]
      if return_item_bank:
        result.append(self.simulator.item_bank)
      if return_administered_items:
        result.append(self.simulator.administered_items[index])
      if return_response_vector:
        result.append(self.simulator.response_vectors[index])
      if return_est_theta:
        result.append(self.simulator.latest_estimations[index])
      if return_rng:
        result.append(self.simulator.rng)

    return tuple(result)


class Initializer(Simulable, ABC):
  """Base class for CAT initializers.

  Initializers are responsible for selecting examinees' initial ability estimates
  before any items are administered.
  """

  def __init__(self) -> None:
    """Initialize an Initializer object."""
    super().__init__()

  @abstractmethod
  def initialize(self, **kwargs: Any) -> float:
    r"""Select an examinee's initial :math:`\theta` value.

    Parameters
    ----------
    **kwargs : dict
        Arguments used by the Initializer implementation.

    Returns
    -------
    float
        Examinee's initial :math:`\theta` value.
    """


class Selector(Simulable, ABC):
  """Base class representing a CAT item selector.

  Selectors are responsible for choosing which item to administer next to an
  examinee based on their current estimated ability and test progress.
  """

  def __init__(self) -> None:
    """Initialize a Selector object."""
    super().__init__()

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
      ordered_items = Selector._sort_by_b(item_bank, est_theta)
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

  @abstractmethod
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
        A list containing the indexes of items that were already administered.
        Default is None.
    est_theta : float or None, optional
        A float containing the current estimated ability. Default is None.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    int or None
        Index of the next item to be applied, or None if there are no more items
        to be presented.
    """


class FiniteSelector(Selector, ABC):
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


class Estimator(Simulable, ABC):
  """Base class for ability estimators.

  Estimators are responsible for computing ability estimates based on examinees'
  responses to administered items.

  Parameters
  ----------
  verbose : bool, optional
      Whether to be verbose during execution. Default is False.
  """

  def __init__(self, verbose: bool = False) -> None:
    """Initialize an Estimator object.

    Parameters
    ----------
    verbose : bool, optional
        Whether to be verbose during execution. Default is False.
    """
    super().__init__()
    self._calls = 0
    self._evaluations = 0
    self._verbose = verbose

  @abstractmethod
  def estimate(
    self,
    index: int | None = None,
    item_bank: ItemBank | None = None,
    administered_items: list[int] | None = None,
    response_vector: list[bool] | None = None,
    est_theta: float | None = None,
  ) -> float:
    r"""Compute the theta value that maximizes the log-likelihood function for the given examinee.

    When this method is used inside a simulator, its arguments are automatically filled.
    Outside of a simulation, the user can also specify the arguments to use the
    Estimator as a standalone object.

    Parameters
    ----------
    index : int or None, optional
        Index of the current examinee in the simulator. Default is None.
    item_bank : ItemBank or None, optional
        An ItemBank containing item parameters. Default is None.
    administered_items : list[int] or None, optional
        A list containing the indexes of items that were already administered.
        Default is None.
    response_vector : list[bool] or None, optional
        A boolean list containing the examinee's answers to the administered items.
        Default is None.
    est_theta : float or None, optional
        A float containing the current estimated ability. Default is None.

    Returns
    -------
    float
        The current estimated ability :math:`\hat\theta`.
    """

  @property
  def calls(self) -> int:
    """Get how many times the estimator has been called to maximize/minimize the log-likelihood function.

    Returns
    -------
    int
        Number of times the estimator has been called to maximize/minimize the
        log-likelihood function.
    """
    return self._calls

  @property
  def evaluations(self) -> int:
    """Get the total number of times the estimator has evaluated the log-likelihood function during its existence.

    Returns
    -------
    int
        Number of function evaluations.
    """
    return self._evaluations

  @property
  def avg_evaluations(self) -> float:
    """Get the average number of function evaluations for all tests the estimator has been used.

    Returns
    -------
    float
        Average number of function evaluations per test.
    """
    return self._evaluations / self._calls


class Stopper(Simulable, ABC):
  """Base class for CAT stopping criteria.

  Stoppers determine when a test should end based on specific criteria such as
  test length, measurement precision, or other conditions.
  """

  def __init__(self) -> None:
    """Initialize a Stopper object."""
    super().__init__()

  @abstractmethod
  def stop(self, index: int | None = None, **kwargs: Any) -> bool:
    """Check whether the test reached its stopping criterion for the given user.

    Parameters
    ----------
    index : int or None, optional
        The index of the current examinee in the simulator. When used within a
        simulation, this parameter is provided automatically. When used standalone,
        other parameters may be provided via kwargs. Default is None.
    **kwargs : dict
        Additional keyword arguments that specific Stopper implementations may require.
        Common arguments include:

        - administered_items: Item parameters or indices that were administered
        - theta: Current ability estimate
        - item_bank: ItemBank for accessing item parameters

    Returns
    -------
    bool
        True if the test met its stopping criterion, False otherwise.
    """


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

  def __init__(
    self,
    item_bank: ItemBank | npt.NDArray[numpy.floating[Any]],
    examinees: int | npt.ArrayLike,
    initializer: Initializer | None = None,
    selector: Selector | None = None,
    estimator: Estimator | None = None,
    stopper: Stopper | None = None,
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
  def initializer(self) -> Initializer | None:
    """Get the initializer used during the simulation.

    Returns
    -------
    Initializer or None
        The initializer used during the simulation.
    """
    return self._initializer

  @property
  def selector(self) -> Selector | None:
    """Get the selector used during the simulation.

    Returns
    -------
    Selector or None
        The selector used during the simulation.
    """
    return self._selector

  @property
  def estimator(self) -> Estimator | None:
    """Get the estimator used during the simulation.

    Returns
    -------
    Estimator or None
        The estimator used during the simulation.
    """
    return self._estimator

  @property
  def stopper(self) -> Stopper | None:
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
    initializer: Initializer | None = None,
    selector: Selector | None = None,
    estimator: Estimator | None = None,
    stopper: Stopper | None = None,
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
    >>> from catsim.stopping import MaxItemStopper
    >>> from catsim.simulation import Simulator
    >>> from catsim.item_bank import ItemBank
    >>> initializer = RandomInitializer()
    >>> selector = MaxInfoSelector()
    >>> estimator = NumericalSearchEstimator()
    >>> stopper = MaxItemStopper(20)
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
