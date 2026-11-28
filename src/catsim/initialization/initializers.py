"""Concrete initialization implementations."""

import math
from typing import override

import numpy
from irtlib import ItemBank

from .base import BaseInitializer


class UniformInitializer(BaseInitializer):
  """Draw each examinee's initial ability from a uniform distribution.

  Parameters
  ----------
  low : float, optional
      Lower bound of the interval. Default is -0.5.
  high : float, optional
      Upper bound of the interval. Default is 0.5.

  Raises
  ------
  TypeError
      If a bound is not a real number.
  ValueError
      If either bound is not finite or ``high <= low``.

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/initialization/index`.
  """

  def __init__(self, low: float = -0.5, high: float = 0.5) -> None:
    if not (math.isfinite(low) and math.isfinite(high)):
      msg = f"low and high must be finite, got low={low}, high={high}"
      raise ValueError(msg)
    if high <= low:
      msg = f"high must be greater than low, got low={low}, high={high}"
      raise ValueError(msg)
    self._low = float(low)
    self._high = float(high)

  @override
  def __repr__(self) -> str:
    """Return the initializer as a constructor call that names every argument."""
    return f"{type(self).__name__}(low={self._low!r}, high={self._high!r})"

  @override
  def initialize(self, *, item_bank: ItemBank, rng: numpy.random.Generator) -> float:
    """Draw an initial ability value between ``low`` and ``high``.

    Parameters
    ----------
    item_bank : ItemBank
        Unused by this initializer.
    rng : numpy.random.Generator
        Generator the value is drawn from.

    Returns
    -------
    float
        An ability value drawn uniformly from the interval.
    """
    return rng.uniform(self._low, self._high)


class NormalInitializer(BaseInitializer):
  """Draw each examinee's initial ability from a normal distribution.

  Parameters
  ----------
  mean : float, optional
      Mean of the distribution. Default is 0.0.
  sd : float, optional
      Standard deviation of the distribution. Default is 1.0.

  Raises
  ------
  TypeError
      If ``mean`` or ``sd`` is not a real number.
  ValueError
      If ``mean`` is not finite, or ``sd`` is not finite and positive.

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/initialization/index`.
  """

  def __init__(self, mean: float = 0.0, sd: float = 1.0) -> None:
    if not math.isfinite(mean):
      msg = f"mean must be finite, got {mean}"
      raise ValueError(msg)
    if not (math.isfinite(sd) and sd > 0):
      msg = f"sd must be positive and finite, got {sd}"
      raise ValueError(msg)
    self._mean = float(mean)
    self._sd = float(sd)

  @override
  def __repr__(self) -> str:
    """Return the initializer as a constructor call that names every argument."""
    return f"{type(self).__name__}(mean={self._mean!r}, sd={self._sd!r})"

  @override
  def initialize(self, *, item_bank: ItemBank, rng: numpy.random.Generator) -> float:
    """Draw an initial ability value from the normal distribution of the given mean and standard deviation.

    Parameters
    ----------
    item_bank : ItemBank
        Unused by this initializer.
    rng : numpy.random.Generator
        Generator the value is drawn from.

    Returns
    -------
    float
        An ability value drawn from the normal distribution.
    """
    return rng.normal(self._mean, self._sd)


class FixedPointInitializer(BaseInitializer):
  """Initialize every examinee's ability at the same fixed point.

  Parameters
  ----------
  start : float
      The starting ability value for every examinee.

  Raises
  ------
  TypeError
      If ``start`` is not a real number.
  ValueError
      If ``start`` is not finite.

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/initialization/index`.
  """

  @override
  def __repr__(self) -> str:
    """Return the initializer as a constructor call that names every argument."""
    return f"{type(self).__name__}(start={self._start!r})"

  def __init__(self, start: float) -> None:
    # math.isfinite raises the TypeError for a non-number, strings included.
    if not math.isfinite(start):
      msg = f"start must be finite, got {start}"
      raise ValueError(msg)
    self._start = float(start)

  @override
  def initialize(self, *, item_bank: ItemBank, rng: numpy.random.Generator) -> float:
    """Return the same ability value that was passed to the constructor.

    Parameters
    ----------
    item_bank : ItemBank
        Unused by this initializer.
    rng : numpy.random.Generator
        Unused by this initializer.

    Returns
    -------
    float
        The fixed ability value that was passed to the constructor.
    """
    return self._start
