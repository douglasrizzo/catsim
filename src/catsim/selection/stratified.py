"""Stratified selector implementations."""

import operator
from abc import abstractmethod

import numpy

from ..exceptions import NoItemsAvailableError
from ..item_bank import ItemBank
from ..typing import FloatArray, IntArray
from .base import BaseSelector, check_test_size, pick_best


class StratifiedSelector(BaseSelector):
  """Abstract base class for selectors that divide the item bank into stage-aligned strata.

  The bank is sorted by a stratification key and split into ``n_strata`` strata, optionally after blocking
  on a location key so that every stratum covers the whole ability range. The test is split into
  ``n_strata`` consecutive stages, and each stage draws from its own stratum the non-administered item
  whose location is closest to the current ability estimate. The strata are the selector's exposure
  control: it applies no exposure cap, and ``exposure_rates`` is ignored.

  Parameters
  ----------
  test_size : int
      Number of items in the test, at least 1. The selector raises ``ValueError`` when asked for an
      item past it, so the stopper must end the test by then.
  n_strata : int or None, optional
      Number of strata (and test stages). Must be between 1 and ``test_size``. Default is None, which
      uses one stratum per test item.

  Notes
  -----
  For full family-level details, see :doc:`/api/techniques/selection/stratified-selector`.
  """

  def __repr__(self) -> str:
    """Return the selector as a constructor call that names every argument."""
    return f"{type(self).__name__}(test_size={self._test_size!r}, n_strata={self._n_strata!r})"

  def __init__(self, test_size: int, n_strata: int | None = None) -> None:
    """Initialize a StratifiedSelector.

    Parameters
    ----------
    test_size : int
        Number of items in the test.
    n_strata : int or None, optional
        Number of strata. Default is None, which uses ``test_size`` strata.

    Raises
    ------
    TypeError
        If ``test_size`` or ``n_strata`` is not an integer.
    ValueError
        If ``test_size`` is smaller than 1 or ``n_strata`` is not between 1 and ``test_size``.
    """
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

    Without blocking, items are sorted by the stratification key and split into ``n_strata`` contiguous
    groups. With blocking :cite:p:`Chang2001`, items are sorted by the blocking key and grouped into
    consecutive blocks of ``n_strata`` items; each block is sorted by the stratification key and its
    :math:`j`-th item goes to stratum :math:`j`, so every stratum spans the range of the blocking key.
    The strata are computed once per item bank.

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
      # a trailing block smaller than n_strata is spread evenly across the strata
      for rank, item in enumerate(block):
        members[rank * self._n_strata // len(block)].append(int(item))
    return [numpy.array(stratum, dtype=numpy.intp) for stratum in members]

  def _stage(self, position: int) -> int:
    """Return the 0-indexed stage of a 0-indexed test position below ``test_size``."""
    return position * self._n_strata // self._test_size

  def select(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,
    rng: numpy.random.Generator | None = None,
    exposure_rates: FloatArray | None = None,  # noqa: ARG002
    response_vector: list[bool] | None = None,  # noqa: ARG002
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
        Random number generator, used only to break ties between equally close items.
    exposure_rates : numpy.ndarray or None, optional
        Ignored: this selector applies no exposure cap.
    response_vector : list[bool] or None, optional
        Unused; accepted for the engine's selector contract.

    Returns
    -------
    int
        Index of the non-administered item in the current stratum whose location is closest to
        ``est_theta``. Ties are broken uniformly at random (ADR-0008).

    Raises
    ------
    NoItemsAvailableError
        If every item in the current stratum has been administered.
    ValueError
        If ``test_size`` items have already been administered, or if items tie for the closest
        location and ``rng`` is None.
    """
    position = len(administered_items)
    if position >= self._test_size:
      msg = (
        f"{type(self).__name__} has test_size={self._test_size}, but {position} items have already been "
        "administered: the test ran past the length its strata were planned for. Give the selector the "
        "stopper's maximum test length as test_size."
      )
      raise ValueError(msg)
    stage = self._stage(position)
    stratum = self.strata(item_bank)[stage]
    candidates = stratum[~numpy.isin(stratum, administered_items)]
    if candidates.size == 0:
      msg = f"There are no more items to be selected from stratum {stage}"
      raise NoItemsAvailableError(msg)

    return pick_best(candidates, -numpy.abs(self._location(item_bank)[candidates] - est_theta), rng)


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
  For full algorithmic details, see :doc:`/api/techniques/selection/a-strat-selector`.
  """

  def _stratification_key(self, item_bank: ItemBank) -> FloatArray:  # noqa: PLR6301
    """Stratify by increasing discrimination."""
    return item_bank.discrimination

  def _location(self, item_bank: ItemBank) -> FloatArray:  # noqa: PLR6301
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
  For full algorithmic details, see :doc:`/api/techniques/selection/a-strat-b-block-selector`.
  """

  def _blocking_key(self, item_bank: ItemBank) -> FloatArray:  # noqa: PLR6301
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
  For full algorithmic details, see :doc:`/api/techniques/selection/max-info-strat-selector`.
  """

  def _stratification_key(self, item_bank: ItemBank) -> FloatArray:  # noqa: PLR6301
    """Stratify by increasing maximum information."""
    return item_bank.max_info_values

  def _location(self, item_bank: ItemBank) -> FloatArray:  # noqa: PLR6301
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
  For full algorithmic details, see :doc:`/api/techniques/selection/max-info-b-block-selector`.
  """

  def _blocking_key(self, item_bank: ItemBank) -> FloatArray:  # noqa: PLR6301
    """Block on the ability of maximum information so every stratum spans the ability range."""
    return item_bank.max_info_thetas
