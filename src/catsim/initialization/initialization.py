"""Concrete initialization implementations."""

from enum import Enum
from typing import Any

from .base import BaseInitializer


class InitializationDistribution(Enum):
  """Distribution to use for ability estimate initialization.

  Attributes
  ----------
  UNIFORM : str
      Uniform distribution for initialization.
  NORMAL : str
      Normal (Gaussian) distribution for initialization.
  """

  UNIFORM = "uniform"
  NORMAL = "normal"


class RandomInitializer(BaseInitializer):
  """Randomly initialize the first estimate of an examinee's ability from a statistical distribution.

  Parameters
  ----------
  dist_type : InitializationDistribution, optional
      Distribution type to use for initialization (uniform or normal).
      Default is UNIFORM.
  dist_params : tuple, optional
      Distribution parameters:
      - For uniform: tuple of (min, max) values (order doesn't matter)
      - For normal: tuple of (mean, std) values (in this exact order)
      Default is (-5, 5).
  """

  def __str__(self) -> str:
    """Get a string representation of the Initializer."""
    return "Random Initializer"

  def __init__(
    self, dist_type: InitializationDistribution = InitializationDistribution.UNIFORM, dist_params: tuple = (-5, 5)
  ) -> None:
    """Initialize a RandomInitializer object.

    Parameters
    ----------
    dist_type : InitializationDistribution, optional
        Distribution to use for ability estimate initialization. Default is UNIFORM.
    dist_params : tuple, optional
        Parameters for the chosen distribution. Default is (-5, 5).
        - For uniform: (min, max) values (order doesn't matter)
        - For normal: (mean, std) values (in this exact order)

    Raises
    ------
    TypeError
        If dist_type is not an InitializationDistribution enum value.
    ValueError
        If dist_params are invalid for the chosen distribution type (e.g., wrong
        length, equal min/max for uniform, non-positive std for normal).
    """
    super().__init__()

    if not isinstance(dist_type, InitializationDistribution):
      msg = "dist_type must be of type InitializationDistribution"
      raise TypeError(msg)

    # Validate distribution parameters
    expected_param_count = 2
    if not isinstance(dist_params, tuple) or len(dist_params) != expected_param_count:
      msg = "dist_params must be a tuple of exactly 2 values"
      raise ValueError(msg)

    if dist_type == InitializationDistribution.UNIFORM:
      min_val, max_val = min(dist_params), max(dist_params)
      if min_val == max_val:
        msg = f"Uniform distribution parameters must have different min and max values, got {dist_params}"
        raise ValueError(msg)
    elif dist_type == InitializationDistribution.NORMAL:
      _, std = dist_params  # mean is not used in validation
      if std <= 0:
        msg = f"Normal distribution standard deviation must be positive, got {std}"
        raise ValueError(msg)

    self._dist_type = dist_type
    self._dist_params = dist_params

  def initialize(self, index: int | None = None, **kwargs: Any) -> float:
    """Generate an initial ability value using the chosen distribution and parameters.

    Parameters
    ----------
    index : int or None, optional
        The index of the current examinee. This parameter is not used by this method.
        Default is None.
    **kwargs : dict
        Additional keyword arguments.

        - rng (numpy.random.Generator): Random number generator used by the object,
          guarantees reproducibility of outputs.

    Returns
    -------
    float
        An ability value generated from the chosen distribution using the specified
        parameters.

    Raises
    ------
    RuntimeError
        If an invalid distribution type is encountered (should not happen after
        validation in __init__).
    """
    (rng,) = self._prepare_args(return_rng=True, index=index, **kwargs)

    if self._dist_type == InitializationDistribution.UNIFORM:
      theta = rng.uniform(min(self._dist_params), max(self._dist_params))
    elif self._dist_type == InitializationDistribution.NORMAL:
      theta = rng.normal(self._dist_params[0], self._dist_params[1])
    else:
      msg = f"Invalid distribution type: {self._dist_type}. This should not happen after validation."
      raise RuntimeError(msg)

    return theta


class FixedPointInitializer(BaseInitializer):
  """Initialize every examinee's ability at the same fixed point.

  This initializer is useful for controlled experiments where you want all
  examinees to start with the same initial ability estimate.

  Parameters
  ----------
  start : float
      The starting ability value for every examinee.
  """

  def __str__(self) -> str:
    """Get a string representation of the Initializer."""
    return "Fixed Point Initializer"

  def __init__(self, start: float) -> None:
    """Initialize a FixedPointInitializer object.

    Parameters
    ----------
    start : float
        The starting ability value for every examinee.

    Raises
    ------
    TypeError
        If start is not a numeric value.
    """
    super().__init__()
    self._start = start

  def initialize(self, index: int | None = None, **kwargs: Any) -> float:  # noqa: ARG002
    """Return the same ability value that was passed to the constructor.

    Parameters
    ----------
    index : int or None, optional
        The index of the current examinee. This parameter is not used by this method.
        Default is None.
    **kwargs : dict
        Additional keyword arguments. Not used by this method.

    Returns
    -------
    float
        The fixed ability value that was passed to the constructor.
    """
    return self._start
