"""Selectors that randomize the choice: random, randomesque and 5-4-3-2-1."""

import operator

from typing_extensions import override

from ..context import StepContext
from .base import BaseSelector, best_pool, remaining


class RandomSelector(BaseSelector):
  """Selector that randomly selects items for application.

  This selector is useful for baseline comparisons or for studying the impact of
  item selection strategies.
  """

  @override
  def __repr__(self) -> str:
    """Return the selector as a constructor call that names every argument."""
    return f"{type(self).__name__}()"

  @override
  def select(self, context: StepContext) -> int:
    """Return a remaining item drawn uniformly at random.

    Parameters
    ----------
    context : StepContext
        The session to select for.

    Returns
    -------
    int
        Index of the selected item.
    """
    return int(context.rng.choice(remaining(context)))


class The54321Selector(BaseSelector):
  r"""Draw each opening item at random from a shrinking bin of the most informative items.

  The 5-4-3-2-1 method of :cite:t:`McBride83`: the first item is drawn at random from the 5 most
  informative items at the current :math:`\hat\theta`, the second from the 4 most informative
  remaining items, and so on, so that from the fifth item on the most informative item is
  administered. Items tied with the least informative item of a bin also join it, so from then on
  ties for the most information are broken uniformly at random. The selector applies no exposure
  cap: ``context.exposure_rates`` is ignored.

  Parameters
  ----------
  initial_bin_size : int, optional
      Number of most informative items the first item is drawn from; each following item draws from
      one fewer. Default is 5.

  Raises
  ------
  TypeError
      If ``initial_bin_size`` is not an integer.
  ValueError
      If ``initial_bin_size`` is smaller than 1.

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/selection/the-54321-selector`.
  """

  @override
  def __repr__(self) -> str:
    """Return the selector as a constructor call that names every argument."""
    return f"{type(self).__name__}(initial_bin_size={self._initial_bin_size!r})"

  def __init__(self, initial_bin_size: int = 5) -> None:
    initial_bin_size = operator.index(initial_bin_size)
    if initial_bin_size < 1:
      msg = f"initial_bin_size must be at least 1, got {initial_bin_size}"
      raise ValueError(msg)
    super().__init__()
    self._initial_bin_size = initial_bin_size

  @property
  def initial_bin_size(self) -> int:
    """Get the number of most informative items the first item is drawn from."""
    return self._initial_bin_size

  @override
  def select(self, context: StepContext) -> int:
    """Return an item drawn uniformly from the bin of the most informative remaining items.

    The bin holds ``initial_bin_size`` items at the first position, one fewer at each following
    position, and at least one, plus every item tied with its least informative item.

    Parameters
    ----------
    context : StepContext
        The session to select for.

    Returns
    -------
    int
        Index of the selected item.
    """
    candidates = remaining(context)
    bin_size = max(1, self._initial_bin_size - len(context.administered_items))
    pool = best_pool(candidates, context.item_bank.information(context.theta)[candidates], bin_size)
    return int(context.rng.choice(pool))


class RandomesqueSelector(BaseSelector):
  """Draw each item at random from the ``bin_size`` most informative remaining items.

  The randomesque method of :cite:t:`Kingsbury89`. Items tied with the least informative item of the
  bin also join it. The selector applies no exposure cap: ``context.exposure_rates`` is ignored.

  Parameters
  ----------
  bin_size : int
      Number of most informative remaining items each item is drawn from, at least 1.

  Raises
  ------
  TypeError
      If ``bin_size`` is not an integer.
  ValueError
      If ``bin_size`` is smaller than 1.
  """

  @override
  def __repr__(self) -> str:
    """Return the selector as a constructor call that names every argument."""
    return f"{type(self).__name__}(bin_size={self._bin_size!r})"

  def __init__(self, bin_size: int) -> None:
    bin_size = operator.index(bin_size)
    if bin_size < 1:
      msg = f"bin_size must be at least 1, got {bin_size}"
      raise ValueError(msg)
    super().__init__()
    self._bin_size = bin_size

  @property
  def bin_size(self) -> int:
    """Get the bin size."""
    return self._bin_size

  @override
  def select(self, context: StepContext) -> int:
    """Return an item drawn uniformly from the ``bin_size`` most informative remaining items.

    The bin also holds every item tied with its least informative item.

    Parameters
    ----------
    context : StepContext
        The session to select for.

    Returns
    -------
    int
        Index of the selected item.
    """
    candidates = remaining(context)
    pool = best_pool(candidates, context.item_bank.information(context.theta)[candidates], self._bin_size)
    return int(context.rng.choice(pool))
