from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .._base import Simulable

if TYPE_CHECKING:
  from ..item_bank import ItemBank


class BaseEstimator(Simulable, ABC):
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
    item_bank: "ItemBank | None" = None,
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
