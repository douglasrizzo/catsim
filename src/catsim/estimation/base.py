from abc import ABC, abstractmethod

from ..item_bank import ItemBank


class BaseEstimator(ABC):
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
    self._calls = 0
    self._last_evaluations = 0
    self._total_evaluations = 0
    self._verbose = verbose

  @abstractmethod
  def estimate(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    response_vector: list[bool],
    est_theta: float,
  ) -> float:
    r"""Compute the theta value that maximizes the log-likelihood function for the given examinee.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    administered_items : list[int]
        A list containing the indexes of items that were already administered.
    response_vector : list[bool]
        A boolean list containing the examinee's answers to the administered items.
    est_theta : float
        The current estimated ability.

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
        Number of function evaluations in the most recent estimate call.
    """
    return self._last_evaluations

  @property
  def total_evaluations(self) -> int:
    """Get the total number of evaluations across the estimator lifetime."""
    return self._total_evaluations

  @property
  def avg_evaluations(self) -> float:
    """Get the average number of function evaluations for all tests the estimator has been used.

    Returns
    -------
    float
        Average number of function evaluations per estimate call.
    """
    if self._calls == 0:
      return 0.0
    return self._total_evaluations / self._calls
