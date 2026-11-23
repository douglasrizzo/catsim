"""Base classes for CAT selectors, and the rules the built-in selectors share.

The module functions are what makes the built-in selectors behave alike, and custom selectors can
use them too: :func:`remaining` and :func:`below_cap` give the items a selector may choose from,
:func:`pick_best` and :func:`best_pool` choose among them under the package's tie-breaking rule
(ADR-0008), and :func:`check_r_max`, :func:`check_test_size`, :func:`require_rng` and
:func:`require_responses` check arguments.
"""

import operator
from abc import ABC, abstractmethod

import numpy
from numpy.random import Generator

from .._typing import FloatArray, IntArray
from ..exceptions import NoItemsAvailableError
from ..item_bank import ItemBank


def remaining(n_items: int, administered_items: list[int]) -> IntArray:
  """Return the indices of the items not yet administered, in ascending order.

  Parameters
  ----------
  n_items : int
      Number of items in the bank.
  administered_items : list[int]
      Indices of the items already administered in the session.

  Returns
  -------
  numpy.ndarray
      Indices of the items of the bank that are not in ``administered_items``, in ascending order.

  Raises
  ------
  NoItemsAvailableError
      If every item has already been administered.
  """
  mask = numpy.ones(n_items, dtype=bool)
  mask[administered_items] = False
  candidates = numpy.flatnonzero(mask)
  if candidates.size == 0:
    msg = "There are no more items to administer: every item in the bank has been administered."
    raise NoItemsAvailableError(msg)
  return candidates


def below_cap(candidates: IntArray, exposure_rates: FloatArray | None, r_max: float) -> IntArray:
  """Return the candidates whose exposure rate is below ``r_max``, or all of them if none is.

  Parameters
  ----------
  candidates : numpy.ndarray
      Item indices to choose from.
  exposure_rates : numpy.ndarray or None
      Run-level exposure rate of every item in the bank. ``None`` treats every item as unexposed.
  r_max : float
      Maximum exposure rate. Candidates at or above it are kept only when every candidate is.

  Returns
  -------
  numpy.ndarray
      The candidates whose exposure rate is below ``r_max``, in their order in ``candidates``, or
      ``candidates`` itself if none is below it.
  """
  if exposure_rates is None:
    return candidates
  allowed = candidates[exposure_rates[candidates] < r_max]
  return allowed if allowed.size > 0 else candidates


def pick_best(candidates: IntArray, scores: FloatArray, rng: Generator | None) -> int:
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
    msg = "rng parameter cannot be None: it is needed to break a tie between equally good candidates"
    raise ValueError(msg)
  return int(rng.choice(best))


def best_pool(candidates: IntArray, scores: FloatArray, size: int) -> IntArray:
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


def check_r_max(r_max: float) -> float:
  """Return ``r_max`` as a float, after checking that it is a usable exposure cap.

  Parameters
  ----------
  r_max : float
      Maximum exposure rate given to a selector.

  Returns
  -------
  float
      ``r_max``.

  Raises
  ------
  ValueError
      If ``r_max`` is not in (0, 1]. With a cap of 0, every item would be at or above it, so the
      selector would fall back to all items and behave as if there were no cap.
  """
  if not 0 < r_max <= 1:
    msg = f"r_max must be in (0, 1], got {r_max}"
    raise ValueError(msg)
  return float(r_max)


def check_test_size(test_size: int) -> int:
  """Return ``test_size`` as a Python int, after checking that it is a usable planned test length.

  Parameters
  ----------
  test_size : int
      Planned number of items in the test, given to a selector whose schedule depends on it. A NumPy
      integer is accepted.

  Returns
  -------
  int
      ``test_size``, as a Python int.

  Raises
  ------
  TypeError
      If ``test_size`` is not an integer.
  ValueError
      If ``test_size`` is smaller than 1.
  """
  test_size = operator.index(test_size)
  if test_size < 1:
    msg = f"test_size must be at least 1, got {test_size}"
    raise ValueError(msg)
  return test_size


def require_rng(rng: Generator | None) -> Generator:
  """Return ``rng``, checking that a selector that draws random numbers was given one.

  Parameters
  ----------
  rng : numpy.random.Generator or None
      Random number generator passed to ``select``.

  Returns
  -------
  numpy.random.Generator
      ``rng``.

  Raises
  ------
  ValueError
      If ``rng`` is None.
  """
  if rng is None:
    msg = "rng parameter cannot be None: this selector draws random numbers"
    raise ValueError(msg)
  return rng


def require_responses(response_vector: list[bool] | None) -> list[bool]:
  """Return ``response_vector``, checking that a selector that scores items by the responses was given them.

  Parameters
  ----------
  response_vector : list[bool] or None
      Responses passed to ``select``.

  Returns
  -------
  list[bool]
      ``response_vector``.

  Raises
  ------
  ValueError
      If ``response_vector`` is None.
  """
  if response_vector is None:
    msg = "response_vector cannot be None: this selector scores items by the responses observed so far"
    raise ValueError(msg)
  return response_vector


class BaseSelector(ABC):
  """Base class representing a CAT item selector.

  Selectors are responsible for choosing which item to administer next to an
  examinee based on their current estimated ability and test progress.
  """

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
        Index of the next item to be applied. The engine treats None like
        ``NoItemsAvailableError``; no built-in selector returns it.

    Raises
    ------
    NoItemsAvailableError
        If no item can be administered, for example because every item has already been
        administered. The engine then ends the session with ``StopReason.NO_ITEMS``.
    """
