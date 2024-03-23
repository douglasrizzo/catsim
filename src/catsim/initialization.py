from enum import Enum
from typing import Any

import numpy

from .simulation import Initializer


class InitializationDistribution(Enum):
  """Distribution to use for ability estimate initialization."""

  UNIFORM = "uniform"
  NORMAL = "normal"


class RandomInitializer(Initializer):
  """Randomly initializes the first estimate of an examinee's ability.

  :param dist_type: either `uniform` or `normal`
  :param dist_params: a tuple containing minimum and maximum values for the uniform distribution (in no particular
  order) or the average and standard deviation values for the normal distribution
  (in this particular order).
  """

  def __str__(self) -> str:
    """Get a string representation of the Initializer."""
    return "Random Initializer"

  def __init__(
    self, dist_type: InitializationDistribution = InitializationDistribution.UNIFORM, dist_params: tuple = (-5, 5)
  ) -> None:
    """Initialize a RandomInitializer object.

    :param dist_type: Distribution to use for ability estimate initialization, defaults to "uniform".
    :type dist_type: InitializationDistribution, optional
    :param dist_params: Parameters for the chosen distribution, defaults to (-5, 5)
    :type dist_params: tuple, optional
    """
    super().__init__()
    assert isinstance(dist_type, InitializationDistribution), "dist_type must be an InitializationDistribution"
    self._dist_type = dist_type
    self._dist_params = dist_params
    self.__rng = numpy.random.default_rng()

  def initialize(self, index: int | None = None, **kwargs: dict[str, Any]) -> float:
    """Generates a value using the chosen distribution and parameters.

    :param index: the index of the current examinee. This parameter is not used by this method.
    :returns: a ability value generated from the chosen distribution using the passed parameters
    """
    if self._dist_type == InitializationDistribution.UNIFORM:
      theta = self.__rng.uniform(min(self._dist_params), max(self._dist_params))
    elif self._dist_type == InitializationDistribution.NORMAL:
      theta = self.__rng.normal(self._dist_params[0], self._dist_params[1])
    return theta


class FixedPointInitializer(Initializer):
  """Initializes every ability at the same point."""

  def __str__(self) -> str:
    """Get a string representation of the Initializer."""
    return "Fixed Point Initializer"

  def __init__(self, start: float) -> None:
    """Initialize a FixedPointInitializer object.

    :param start: the starting point for every examinee.
    """
    super().__init__()
    self._start = start

  def initialize(self, index: int | None = None, **kwargs: dict[str, Any]) -> float:
    """Returns the same ability value that was passed to the constructor of the initializer.

    :param index: the index of the current examinee. This parameter is not used by this method.
    :returns: the same ability value that was passed to the constructor of the initializer
    """
    return self._start
