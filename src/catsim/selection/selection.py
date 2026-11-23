"""Concrete selector implementations."""

import numpy

from .. import irt
from .._typing import FloatArray
from ..exceptions import NoItemsAvailableError
from ..item_bank import ItemBank
from .base import BaseSelector


class MaxInfoSelector(BaseSelector):
  """Select the non-administered item with maximum information at the current theta estimate.

  Parameters
  ----------
  r_max : float, optional
      Maximum exposure rate for items. Items with exposure rates >= r_max will not
      be selected unless no other items are available. Default is 1 (no restriction).

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/selection/max-info-selector`.
  """

  def __init__(self, r_max: float = 1) -> None:
    """Initialize a MaxInfoSelector object.

    Parameters
    ----------
    r_max : float, optional
        Maximum exposure rate for items. Default is 1.

    Raises
    ------
    ValueError
        If r_max is not between 0 and 1.
    """
    if not 0 <= r_max <= 1:
      msg = f"r_max must be between 0 and 1, got {r_max}"
      raise ValueError(msg)

    super().__init__()
    self._r_max = float(r_max)

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "Maximum Information Selector"

  @property
  def r_max(self) -> float:
    """Return the maximum exposure rate for items the selector accepts.

    Returns
    -------
    float
        Maximum exposure rate for items the selector accepts.
    """
    return self._r_max

  def select(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,
    rng: numpy.random.Generator | None = None,
    exposure_rates: FloatArray | None = None,
    response_vector: list[bool] | None = None,  # noqa: ARG002
  ) -> int | None:
    """Return the index of the next item to be administered.

    Among the non-administered items below ``r_max`` (or all non-administered items, if every one
    is at or above it), returns the most informative at ``est_theta``. Ties are broken uniformly at
    random (ADR-0008).

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    administered_items : list[int]
        A list containing the indexes of items that were already administered.
    est_theta : float
        The current estimated ability.
    rng : numpy.random.Generator or None, optional
        Random number generator, used only to break ties between equally informative items.
    exposure_rates : numpy.ndarray or None, optional
        Run-level exposure-rate snapshot. ``None`` treats every item as unexposed.

    Returns
    -------
    int
        Index of the next item to be applied.

    Raises
    ------
    NoItemsAvailableError
        If every item has already been administered.
    ValueError
        If items tie for the most information and ``rng`` is None.
    """
    candidates = self._exposure_filtered_candidates(item_bank.n_items, administered_items, exposure_rates, self._r_max)
    return self._pick_best(candidates, item_bank.information(est_theta)[candidates], rng)


class UrrySelector(BaseSelector):
  """Select the remaining item whose difficulty parameter is closest to the current ability estimate.

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/selection/urry-selector`.
  """

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "Urry Selector"

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

    Returns the non-administered item whose difficulty is closest to ``est_theta``. Ties are broken
    uniformly at random (ADR-0008).

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

    Returns
    -------
    int
        Index of the next item to be applied.

    Raises
    ------
    NoItemsAvailableError
        If every item has already been administered.
    ValueError
        If items tie for the closest difficulty and ``rng`` is None.
    """
    candidates = self._exposure_filtered_candidates(item_bank.n_items, administered_items, None, 1.0)
    return self._pick_best(candidates, -numpy.abs(item_bank.difficulty[candidates] - est_theta), rng)


class IntervalInfoSelector(BaseSelector):
  r"""Select the item whose information, integrated over an interval around the estimate, is largest.

  By default the interval is the confidence interval of the interval information criterion
  :cite:p:`Veerkamp1997`, which narrows as the test information grows:

  .. math::

      \arg\max_{i \in R} \int_{\theta_L}^{\theta_U} I_i(\theta)\, d\theta, \qquad
      \theta_{L, U} = \hat\theta \mp \frac{\lambda}{\sqrt{I(\hat\theta)}}, \qquad
      \lambda = \Phi^{-1}\left(\frac{1 + \gamma}{2}\right)

  where :math:`I(\hat\theta)` is the information of the administered items and :math:`\gamma` the
  confidence level. A fixed half-width :math:`\delta` can be used instead.

  Parameters
  ----------
  interval : float or None, optional
      Fixed half-width :math:`\delta` of the interval. Default is None, which uses the confidence
      interval above.
  confidence : float, optional
      Confidence level :math:`\gamma` of the default interval. Default is 0.95.

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/selection/interval-info-selector`.
  """

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "Interval Information Selector"

  def __init__(self, interval: float | None = None, confidence: float = 0.95) -> None:
    r"""Initialize an IntervalInfoSelector object.

    Parameters
    ----------
    interval : float or None, optional
        Fixed half-width of the interval. Default is None, which uses a confidence interval of
        :math:`\hat\theta`.
    confidence : float, optional
        Confidence level of the default interval. Default is 0.95.

    Raises
    ------
    ValueError
        If ``interval`` is not positive or ``confidence`` is not between 0 and 1.
    """
    if interval is not None and interval <= 0:
      msg = f"interval must be positive, got {interval}"
      raise ValueError(msg)
    if not 0 < confidence < 1:
      msg = f"confidence must be between 0 and 1, got {confidence}"
      raise ValueError(msg)

    super().__init__()
    self._interval = interval
    self._confidence = confidence

  @property
  def interval(self) -> float | None:
    """Get the fixed half-width of the interval, or None when the confidence interval is used."""
    return self._interval

  @property
  def confidence(self) -> float:
    """Get the confidence level of the default interval."""
    return self._confidence

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

    Ties in the integrated information are broken uniformly at random (ADR-0008).

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    administered_items : list[int]
        A list containing the indexes of items that were already administered.
    est_theta : float
        The current estimated ability.
    rng : numpy.random.Generator or None, optional
        Random number generator, used only to break ties between equally good items.

    Returns
    -------
    int
        Index of the next item to be applied.

    Raises
    ------
    NoItemsAvailableError
        If every item has already been administered.
    ValueError
        If items tie for the largest integrated information and ``rng`` is None.
    """
    candidates = numpy.setdiff1d(numpy.arange(item_bank.n_items), administered_items)
    if candidates.size == 0:
      msg = "There are no more items to apply."
      raise NoItemsAvailableError(msg)

    if self._interval is None:
      # infinite before any item is administered, since there is no information yet
      lower, upper = irt.confidence_interval(est_theta, item_bank.get_items(administered_items), self._confidence)
    else:
      lower, upper = est_theta - self._interval, est_theta + self._interval

    information_integral = irt.inf_integral(lower, upper, item_bank.items[candidates])
    return self._pick_best(candidates, information_integral, rng)
