"""Base class for CAT selectors, and the rules the built-in selectors share.

The module functions are what makes the built-in selectors behave alike, and custom selectors can
use them too: :func:`remaining` and :func:`below_cap` give the items a selector may choose from,
:func:`pick_best` and :func:`best_pool` choose among them under the package's tie-breaking rule,
and :func:`check_r_max` and :func:`check_test_size` check constructor arguments.
"""

import operator
from abc import ABC, abstractmethod

import numpy
from numpy.random import Generator

from ..context import StepContext
from ..exceptions import NoItemsAvailableError
from ..typing import FloatArray, IntArray


def remaining(context: StepContext) -> IntArray:
  """Return the indices of the items a selector may choose from, in ascending order.

  Parameters
  ----------
  context : StepContext
      The session to select for.

  Returns
  -------
  numpy.ndarray
      Indices of the items of the bank that are neither administered nor excluded, in ascending
      order.

  Raises
  ------
  NoItemsAvailableError
      If every item has been administered or excluded.
  """
  mask = numpy.ones(context.item_bank.n_items, dtype=bool)
  # numpy reads a tuple index as one index per axis, so the tuples go in as arrays
  mask[numpy.asarray(context.administered_items, dtype=numpy.intp)] = False
  mask[numpy.asarray(context.excluded_items, dtype=numpy.intp)] = False
  candidates = numpy.flatnonzero(mask)
  if candidates.size == 0:
    msg = "There are no more items to administer: every item in the bank has been administered or excluded."
    raise NoItemsAvailableError(msg)
  return candidates


def below_cap(candidates: IntArray, exposure_rates: FloatArray | None, r_max: float) -> IntArray:
  """Return the candidate items whose exposure rate is below ``r_max``, or all of them if none is.

  Parameters
  ----------
  candidates : numpy.ndarray
      Item indices to choose from.
  exposure_rates : numpy.ndarray or None
      Run-level exposure rate of every item in the bank. ``None`` treats every item as unexposed.
  r_max : float
      Maximum exposure rate. Candidates at or above it are kept only when every candidate is. 1
      applies no cap.

  Returns
  -------
  numpy.ndarray
      The candidates whose exposure rate is below ``r_max``, in their order in ``candidates``, or
      ``candidates`` itself if none is below it or ``r_max`` is 1.
  """
  # As in catR, a cap of 1 is no cap: an item that every earlier examinee received has a rate of
  # exactly 1, which the strict comparison below would pass over.
  if exposure_rates is None or r_max >= 1:
    return candidates
  allowed = candidates[exposure_rates[candidates] < r_max]
  return allowed if allowed.size > 0 else candidates


def pick_best(candidates: IntArray, scores: FloatArray, rng: Generator) -> int:
  """Return the candidate item with the highest score, breaking exact ties uniformly at random.

  This is the package's single tie-breaking rule: the candidates whose score equals the best score
  exactly are tied, and one of them is drawn uniformly with ``rng``. With a single best candidate,
  ``rng`` is not used, so untied selections consume no random numbers.

  Parameters
  ----------
  candidates : numpy.ndarray
      Item indices to choose from.
  scores : numpy.ndarray
      One score per candidate; higher is better. Selectors that minimize a quantity, such as the
      distance to the ability estimate, pass its negation.
  rng : numpy.random.Generator
      Random number generator used to break ties.

  Returns
  -------
  int
      Index of the selected item.

  Raises
  ------
  ValueError
      If any score is NaN.
  """
  if numpy.isnan(scores).any():
    msg = "item scores contain NaN"
    raise ValueError(msg)
  best = candidates[scores == numpy.max(scores)]
  if best.size == 1:
    return int(best[0])
  return int(rng.choice(best))


def best_pool(candidates: IntArray, scores: FloatArray, size: int) -> IntArray:
  """Return the candidate items scoring at least the ``size``-th best score, best first.

  Every candidate tied with the ``size``-th best score joins the pool, so the pool holds more than
  ``size`` items only when items tie at the cutoff. Tied candidates keep
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


class BaseSelector(ABC):
  """Base class for item selectors.

  A selector chooses the next item to administer from what a :class:`~catsim.StepContext` tells it
  about the session. A custom selector behaves like the built-in ones when it draws its candidate items
  with :func:`remaining`, applies its exposure cap with :func:`below_cap`, and chooses with
  :func:`pick_best` or :func:`best_pool`, which break ties the same way.

  A selector that passes a changed context to another adds to its excluded items, never replaces
  them. A selector holds its configuration and caches derived from the item bank, never
  per-session state: one selector serves every session of a run, which an application may
  interleave. The :doc:`/extending` guide writes one.
  """

  @abstractmethod
  def select(self, context: StepContext) -> int:
    """Return the index of the next item to administer.

    Parameters
    ----------
    context : StepContext
        The session to select for.

    Returns
    -------
    int
        Index of an item of ``context.item_bank`` that is neither administered nor excluded.

    Raises
    ------
    NoItemsAvailableError
        If no item can be administered, for example because every item has been administered or
        excluded. The engine then ends the session with ``StopReason.NO_ITEMS``.
    """
