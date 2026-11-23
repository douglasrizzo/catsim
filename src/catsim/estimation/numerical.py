"""Numerical estimation methods for ability estimation."""

from scipy.optimize import minimize_scalar

from .. import cat, irt
from ..irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED
from ..item_bank import ItemBank
from .base import BaseEstimator


class NumericalSearchEstimator(BaseEstimator):
  """Estimate ability by numerically maximizing the IRT log-likelihood.

  The maximum is searched for with SciPy's bounded scalar minimizer on
  ``[THETA_MIN_EXTENDED, THETA_MAX_EXTENDED]``.

  Parameters
  ----------
  tol : float, optional
      Absolute tolerance of the bounded search, in the ability scale. Default is 1e-6.
  dodd : bool, optional
      Whether to employ Dodd's estimation heuristic :cite:p:`Dod90` when the response vector
      only has one kind of response (all correct or all incorrect, see
      :py:func:`catsim.cat.dodd`). Default is True.
  verbose : bool, optional
      Whether to print detailed information during optimization. Default is False.

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/estimation/numerical-search-estimator`.
  """

  def __str__(self) -> str:
    """Return a string representation of the estimator."""
    return "Numerical Search Estimator"

  def __init__(
    self,
    tol: float = 1e-6,
    dodd: bool = True,
    verbose: bool = False,
  ) -> None:
    """Initialize the estimator.

    Parameters
    ----------
    tol : float, optional
        Absolute tolerance of the bounded search, in the ability scale. Default is 1e-6.
    dodd : bool, optional
        Whether to use Dodd's estimation heuristic for edge cases (all correct or
        all incorrect responses). Default is True.
    verbose : bool, optional
        Whether to print detailed information during optimization. Default is False.

    Raises
    ------
    ValueError
        If ``tol`` is not positive.
    """
    if not tol > 0:
      msg = f"tol must be positive, got {tol}"
      raise ValueError(msg)
    super().__init__(verbose)

    self._tol = float(tol)
    self._dodd = dodd

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

    Raises
    ------
    ValueError
        If required parameters are missing, if ``response_vector`` and ``administered_items`` have
        different lengths, or if ``response_vector`` contains non-Boolean elements.
    """
    self._require_inputs(item_bank, administered_items, response_vector)

    self._calls += 1
    self._last_evaluations = 0

    summarized_answers = set(response_vector)

    # enter here if examinee has only answered correctly or incorrectly
    if len(summarized_answers) == 1:
      answer = summarized_answers.pop()

      # if the estimator was initialized with dodd = True,
      # use Dodd's estimation heuristic to return a theta value
      if self._dodd:
        candidate_theta = cat.dodd(est_theta, item_bank, answer)

      # otherwise, return positive or negative infinity,
      # in accordance with the definition of the MLE
      elif answer:
        candidate_theta = float("inf")
      else:
        candidate_theta = float("-inf")

      return candidate_theta

    res = minimize_scalar(
      irt.negative_log_likelihood,
      bounds=(THETA_MIN_EXTENDED, THETA_MAX_EXTENDED),
      method="bounded",
      args=(response_vector, item_bank.get_items(administered_items)),
      # SciPy's bounded method takes its absolute tolerance only through options
      options={"xatol": self._tol},
    )
    self._last_evaluations = res.nfev
    self._total_evaluations += self._last_evaluations

    if self._verbose:
      print(f"{self._last_evaluations} evaluations")

    return res.x

  @property
  def dodd(self) -> bool:
    """Whether Dodd's estimation heuristic :cite:p:`Dod90` will be used by the estimator.

    Dodd's method is used when the response vector is composed solely of correct or
    incorrect answers, to prevent maximum likelihood methods from returning -infinity
    or +infinity.

    Returns
    -------
    bool
        Boolean value indicating if Dodd's method will be used or not.

    See Also
    --------
    catsim.cat.dodd : Implementation of Dodd's estimation heuristic.
    """
    return self._dodd
