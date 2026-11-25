"""Stratified selector implementations."""

import operator
from abc import abstractmethod

import numpy

from ..compat import override
from ..context import StepContext
from ..exceptions import NoItemsAvailableError
from ..item_bank import ItemBank
from ..typing import FloatArray, IntArray
from .base import BaseSelector, check_test_size, pick_best, remaining


class StratifiedSelector(BaseSelector):
  """Abstract base class for selectors that divide the item bank into stage-aligned strata.

  The bank is sorted by a stratification key and split into ``n_strata`` strata, optionally after blocking
  on a location key so that every stratum covers the whole ability range. The test is split into
  ``n_strata`` consecutive stages, and each stage draws from its own stratum the non-administered item
  whose location is closest to the current ability estimate. The strata are its exposure control:
  it applies no exposure cap and ignores ``context.exposure_rates``.

  Parameters
  ----------
  test_size : int
      Number of items in the test, at least 1. The selector raises ``ValueError`` when asked for an
      item past it, so the stopper must end the test by then.
  n_strata : int or None, optional
      Number of strata (and test stages). Must be between 1 and ``test_size``. Default is None, which
      uses one stratum per test item.

  Raises
  ------
  TypeError
      If ``test_size`` or ``n_strata`` is not an integer.
  ValueError
      If ``test_size`` is smaller than 1 or ``n_strata`` is not between 1 and ``test_size``.

  Notes
  -----
  For full family-level details, see :doc:`/api/techniques/selection/stratified-selector`.
  """

  @override
  def __repr__(self) -> str:
    """Return the selector as a constructor call that names every argument."""
    return f"{type(self).__name__}(test_size={self._test_size!r}, n_strata={self._n_strata!r})"

  def __init__(self, test_size: int, n_strata: int | None = None) -> None:
    super().__init__()
    self._test_size = check_test_size(test_size)
    n_strata = self._test_size if n_strata is None else operator.index(n_strata)
    if not 1 <= n_strata <= self._test_size:
      msg = f"n_strata must be between 1 and test_size ({test_size}), got {n_strata}"
      raise ValueError(msg)
    self._n_strata = n_strata
    self._strata: list[IntArray] | None = None
    self._strata_item_bank: ItemBank | None = None

  @property
  def test_size(self) -> int:
    """Get the number of items in the test."""
    return self._test_size

  @property
  def n_strata(self) -> int:
    """Get the number of strata (and test stages)."""
    return self._n_strata

  @abstractmethod
  def _stratification_key(self, item_bank: ItemBank) -> FloatArray:
    """Return the per-item values whose ascending order assigns items to strata."""

  @abstractmethod
  def _location(self, item_bank: ItemBank) -> FloatArray:
    """Return the per-item ability-scale locations matched against the current estimate."""

  def _blocking_key(self, item_bank: ItemBank) -> FloatArray | None:  # noqa: ARG002, PLR6301
    """Return the per-item values to block on before stratifying, or None for no blocking."""
    return None

  def strata(self, item_bank: ItemBank) -> list[IntArray]:
    """Return the item indices of each stratum, from the first test stage to the last.

    The strata ascend in the stratification key; with blocking, every stratum spans the range of the
    blocking key. The strata are computed once per item bank.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.

    Returns
    -------
    list[numpy.ndarray]
        One array of item indices per stratum.
    """
    if self._strata is None or self._strata_item_bank is not item_bank:
      self._strata = self._build_strata(item_bank)
      self._strata_item_bank = item_bank
    return self._strata

  def _build_strata(self, item_bank: ItemBank) -> list[IntArray]:
    key = self._stratification_key(item_bank)
    blocking_key = self._blocking_key(item_bank)
    if blocking_key is None:
      return numpy.array_split(numpy.argsort(key, kind="stable"), self._n_strata)

    by_block_key = numpy.argsort(blocking_key, kind="stable")
    members: list[list[int]] = [[] for _ in range(self._n_strata)]
    for start in range(0, item_bank.n_items, self._n_strata):
      block = by_block_key[start : start + self._n_strata]
      block = block[numpy.argsort(key[block], kind="stable")]
      # a trailing block smaller than n_strata fills the first strata, so the strata are sized like array_split's
      for rank, item in enumerate(block):
        members[rank].append(int(item))
    return [numpy.array(stratum, dtype=numpy.intp) for stratum in members]

  def _stage(self, position: int) -> int:
    """Return the 0-indexed stage of a 0-indexed test position below ``test_size``."""
    # the first test_size % n_strata stages get a position more, like the first n_items % n_strata strata get an item
    # more, so no stage of a test of at most n_items items needs more items than its stratum holds
    short, n_long = divmod(self._test_size, self._n_strata)
    long_positions = n_long * (short + 1)
    if position < long_positions:
      return position // (short + 1)
    return n_long + (position - long_positions) // short

  @override
  def select(self, context: StepContext) -> int:
    """Return the available item of the current stage's stratum whose location is closest to the estimate.

    Parameters
    ----------
    context : StepContext
        The session to select for.

    Returns
    -------
    int
        Index of the selected item.

    Raises
    ------
    NoItemsAvailableError
        If every item in the current stratum has been administered or excluded.
    ValueError
        If ``test_size`` items have already been administered.
    """
    position = len(context.administered_items)
    if position >= self._test_size:
      msg = (
        f"{type(self).__name__} has test_size={self._test_size}, but {position} items have already been "
        "administered: the test ran past the length its strata were planned for. Give the selector the "
        "stopper's maximum test length as test_size."
      )
      raise ValueError(msg)
    stage = self._stage(position)
    stratum = self.strata(context.item_bank)[stage]
    candidates = stratum[numpy.isin(stratum, remaining(context))]
    if candidates.size == 0:
      msg = f"There are no more items to be selected from stratum {stage}"
      raise NoItemsAvailableError(msg)

    location = self._location(context.item_bank)[candidates]
    return pick_best(candidates, -numpy.abs(location - context.theta), context.rng)


class AStratSelector(StratifiedSelector):
  r"""Implementation of the :math:`\alpha`-stratified selector :cite:p:`Chang1999`.

  Parameters
  ----------
  test_size : int
      The number of items the test contains.
  n_strata : int or None, optional
      Number of strata. Default is None, which uses ``test_size`` strata.

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/selection/stratified-selector`.
  """

  @override
  def _stratification_key(self, item_bank: ItemBank) -> FloatArray:
    """Stratify by increasing discrimination."""
    return item_bank.discrimination

  @override
  def _location(self, item_bank: ItemBank) -> FloatArray:
    """Match item difficulty to the ability estimate."""
    return item_bank.difficulty


class AStratBBlockSelector(AStratSelector):
  r"""Implementation of the :math:`\alpha`-stratified selector with :math:`b` blocking :cite:p:`Chang2001`.

  Parameters
  ----------
  test_size : int
      The number of items the test contains.
  n_strata : int or None, optional
      Number of strata. Default is None, which uses ``test_size`` strata.

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/selection/stratified-selector`.
  """

  @override
  def _blocking_key(self, item_bank: ItemBank) -> FloatArray:
    """Block on item difficulty so every stratum spans the difficulty range."""
    return item_bank.difficulty


class MaxInfoStratSelector(StratifiedSelector):
  """Implementation of the maximum information stratification (MIS) selector :cite:p:`Barrada2006`.

  Parameters
  ----------
  test_size : int
      The number of items the test contains.
  n_strata : int or None, optional
      Number of strata. Default is None, which uses ``test_size`` strata.

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/selection/stratified-selector`.
  """

  @override
  def _stratification_key(self, item_bank: ItemBank) -> FloatArray:
    """Stratify by increasing maximum information."""
    return item_bank.max_info_values

  @override
  def _location(self, item_bank: ItemBank) -> FloatArray:
    """Match the ability at which each item is most informative to the ability estimate."""
    return item_bank.max_info_thetas


class MaxInfoBBlockSelector(MaxInfoStratSelector):
  """Implementation of the maximum information stratification with blocking (MIS-B) selector :cite:p:`Barrada2006`.

  Parameters
  ----------
  test_size : int
      The number of items the test contains.
  n_strata : int or None, optional
      Number of strata. Default is None, which uses ``test_size`` strata.

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/selection/stratified-selector`.
  """

  @override
  def _blocking_key(self, item_bank: ItemBank) -> FloatArray:
    """Block on the ability of maximum information so every stratum spans the ability range."""
    return item_bank.max_info_thetas
