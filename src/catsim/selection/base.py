"""Base class for CAT selectors."""

from abc import ABC, abstractmethod

import numpy
from numpy.random import Generator

from .._typing import FloatArray, IntArray
from ..exceptions import NoItemsAvailableError
from ..item_bank import ItemBank


class BaseSelector(ABC):
  """Base class representing a CAT item selector.

  Selectors are responsible for choosing which item to administer next to an
  examinee based on their current estimated ability and test progress.
  """

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
  def _exposure_filtered_candidates(
    n_items: int,
    administered_items: list[int],
    exposure_rates: FloatArray | None,
    r_max: float,
  ) -> IntArray:
    """Return non-administered item indices, preferring items below the exposure cap.

    Parameters
    ----------
    n_items : int
        Number of items in the bank.
    administered_items : list[int]
        Indices of items already administered in the session.
    exposure_rates : numpy.ndarray or None
        Run-level exposure-rate snapshot. ``None`` treats every item as unexposed.
    r_max : float
        Maximum exposure rate; items at or above it are only used when no other item remains.

    Returns
    -------
    numpy.ndarray
        Indices of the non-administered items below ``r_max``, or of all non-administered
        items if every one of them is at or above ``r_max``.

    Raises
    ------
    NoItemsAvailableError
        If every item has already been administered.
    """
    candidate_mask = numpy.ones(n_items, dtype=bool)
    candidate_mask[administered_items] = False
    all_candidates = numpy.flatnonzero(candidate_mask)
    if all_candidates.size == 0:
      msg = "There are no more items to apply."
      raise NoItemsAvailableError(msg)
    if exposure_rates is None:
      return all_candidates
    low_exposure = numpy.flatnonzero(candidate_mask & (exposure_rates < r_max))
    return low_exposure if low_exposure.size > 0 else all_candidates

  @staticmethod
  def _pick_best(
    candidates: IntArray,
    scores: FloatArray,
    rng: Generator | None,
  ) -> int:
    """Return the candidate with the highest score, breaking exact ties uniformly at random.

    This is the package's single tie-breaking rule (ADR-0008), the one catR uses: the candidates
    whose score equals the best score exactly are tied, and one of them is drawn uniformly with
    ``rng``. With a single best candidate, ``rng`` is not used, so untied selections consume no
    random numbers.

    Parameters
    ----------
    candidates : numpy.ndarray
        Item indices to choose from.
    scores : numpy.ndarray
        One score per candidate; higher is better. Selectors that minimize a quantity, such as the
        distance to the ability estimate, pass its negation.
    rng : numpy.random.Generator or None
        Random number generator used to break ties.

    Returns
    -------
    int
        Index of the selected item.

    Raises
    ------
    ValueError
        If any score is NaN, or if several candidates tie for the best score and ``rng`` is None.
    """
    if numpy.isnan(scores).any():
      msg = "item scores contain NaN"
      raise ValueError(msg)
    best = candidates[scores == scores.max()]
    if best.size == 1:
      return int(best[0])
    if rng is None:
      msg = "rng parameter cannot be None: it is needed to break a tie between equally good items"
      raise ValueError(msg)
    return int(rng.choice(best))

  @staticmethod
  def _best_pool(
    candidates: IntArray,
    scores: FloatArray,
    size: int,
  ) -> IntArray:
    """Return the candidates scoring at least the ``size``-th best score, best first.

    As in catR (ADR-0008), every candidate tied with the ``size``-th best score joins the pool, so
    the pool holds more than ``size`` items only when items tie at the cutoff. Tied candidates keep
    their order in ``candidates``.

    Parameters
    ----------
    candidates : numpy.ndarray
        Item indices to choose from.
    scores : numpy.ndarray
        One score per candidate; higher is better.
    size : int
        Number of best candidates the pool should hold, before ties at the cutoff are added.

    Returns
    -------
    numpy.ndarray
        Item indices of the pool, in descending order of score.

    Raises
    ------
    ValueError
        If ``size`` is smaller than 1 or any score is NaN.
    """
    if size < 1:
      msg = f"the pool size must be at least 1, got {size}"
      raise ValueError(msg)
    if numpy.isnan(scores).any():
      msg = "item scores contain NaN"
      raise ValueError(msg)
    order = numpy.argsort(-scores, kind="stable")
    ranked_scores = scores[order]
    cutoff = ranked_scores[min(size, ranked_scores.size) - 1]
    return candidates[order[ranked_scores >= cutoff]]

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
      ordered_items = BaseSelector._sort_by_b(item_bank, est_theta)
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

  @staticmethod
  def _require_item_bank(item_bank: ItemBank | None) -> ItemBank:
    if item_bank is None:
      msg = "item_bank parameter cannot be None"
      raise ValueError(msg)
    return item_bank

  @staticmethod
  def _require_administered_items(administered_items: list[int] | None) -> list[int]:
    if administered_items is None:
      msg = "administered_items parameter cannot be None"
      raise ValueError(msg)
    return administered_items

  @staticmethod
  def _require_est_theta(est_theta: float | None) -> float:
    if est_theta is None:
      msg = "est_theta parameter cannot be None"
      raise ValueError(msg)
    return est_theta

  @abstractmethod
  def select(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,
    rng: Generator | None = None,
    exposure_rates: FloatArray | None = None,
    response_vector: list[bool] | None = None,
  ) -> int | None:
    """Return the index of the next item to be administered.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    administered_items : list[int]
        A list containing the indexes of items that were already administered.
    est_theta : float
        The current estimated ability.
    rng : numpy.random.Generator or None, optional
        Random number generator for selectors that rely on randomness.
    exposure_rates : numpy.ndarray or None, optional
        Explicit run-level exposure-rate snapshot.
    response_vector : list[bool] or None, optional
        Responses observed so far in the current session. The engine always passes
        it, so selectors can use the response history without relying on hidden
        engine state. Selectors that need it, such as MEISelector and MLWISelector,
        raise ValueError when it is None.

    Returns
    -------
    int or None
        Index of the next item to be applied. None ends the session on the selector's
        own terms, with stop reason ``"selector_stopped"``; no built-in selector returns it.

    Raises
    ------
    NoItemsAvailableError
        If no item can be administered, for example because every item has already been
        administered. The engine then ends the session with stop reason
        ``"item_bank_exhausted"``.
    """


class FiniteSelector(BaseSelector, ABC):
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
