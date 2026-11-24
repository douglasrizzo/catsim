"""Metrics that score a simulation study: how accurate the ability estimates are and how much the tests overlap."""

import math

import numpy
import numpy.typing as npt

from .typing import FloatArray


def bias(
  actual: npt.ArrayLike,
  predicted: npt.ArrayLike,
) -> float:
  r"""Return the mean signed error of the ability estimates :cite:p:`Chang2001`.

  .. math:: \mathrm{Bias} = \frac{1}{N} \sum_{j=1}^{N} (\hat{\theta}_j - \theta_j)

  where :math:`\theta_j` is examinee :math:`j`'s true ability, :math:`\hat{\theta}_j` their estimate
  and :math:`N` the number of examinees. A positive bias means the estimates overestimate ability on
  average, a negative one that they underestimate it.

  Parameters
  ----------
  actual : array-like
      The true abilities.
  predicted : array-like
      The estimated abilities, one per true ability.

  Returns
  -------
  float
      The bias.

  Raises
  ------
  ValueError
      If ``actual`` and ``predicted`` differ in length.

  Examples
  --------
  >>> from catsim.metrics import bias
  >>> bias([1.0, -1.0], [0.0, 6.0])
  3.0
  """
  actual = numpy.asarray(actual)
  predicted = numpy.asarray(predicted)

  if len(actual) != len(predicted):
    msg = "actual and predicted vectors need to be the same size"
    raise ValueError(msg)
  return float(numpy.mean(predicted - actual))


def mse(
  actual: npt.ArrayLike,
  predicted: npt.ArrayLike,
) -> float:
  r"""Return the mean squared error of the ability estimates :cite:p:`Chang2001`.

  .. math:: \mathrm{MSE} = \frac{1}{N} \sum_{j=1}^{N} (\hat{\theta}_j - \theta_j)^2

  where :math:`\theta_j` is examinee :math:`j`'s true ability, :math:`\hat{\theta}_j` their estimate
  and :math:`N` the number of examinees. Lower values mean more accurate estimates; the MSE is the
  squared :func:`bias` plus the variance of the errors.

  Parameters
  ----------
  actual : array-like
      The true abilities.
  predicted : array-like
      The estimated abilities, one per true ability.

  Returns
  -------
  float
      The mean squared error.

  Raises
  ------
  ValueError
      If ``actual`` and ``predicted`` differ in length.

  Examples
  --------
  >>> from catsim.metrics import mse
  >>> mse([1.0, -1.0], [0.0, 6.0])
  25.0
  """
  actual = numpy.asarray(actual)
  predicted = numpy.asarray(predicted)

  if len(actual) != len(predicted):
    msg = "Actual and predicted vectors need to be the same size"
    raise ValueError(msg)
  diff = predicted - actual
  return float(numpy.mean(diff * diff))


def rmse(
  actual: npt.ArrayLike,
  predicted: npt.ArrayLike,
) -> float:
  r"""Return the root mean squared error of the ability estimates :cite:p:`Bar10`.

  .. math:: \mathrm{RMSE} = \sqrt{\frac{1}{N} \sum_{j=1}^{N} (\hat{\theta}_j - \theta_j)^2}

  where :math:`\theta_j` is examinee :math:`j`'s true ability, :math:`\hat{\theta}_j` their estimate
  and :math:`N` the number of examinees. It is the square root of :func:`mse`, in the units of the
  ability scale.

  Parameters
  ----------
  actual : array-like
      The true abilities.
  predicted : array-like
      The estimated abilities, one per true ability.

  Returns
  -------
  float
      The root mean squared error.

  Raises
  ------
  ValueError
      If ``actual`` and ``predicted`` differ in length.

  Examples
  --------
  >>> from catsim.metrics import rmse
  >>> rmse([1.0, -1.0], [0.0, 6.0])
  5.0
  """
  return math.sqrt(mse(actual, predicted))


def overlap_rate(exposure_rates: FloatArray, test_size: int) -> float:
  r"""Return the test overlap rate: the expected share of items two examinees' tests have in common :cite:p:`Bar10`.

  .. math:: T = \frac{M}{n} S_r^2 + \frac{n}{M}

  where :math:`M` is the bank size, :math:`n` the test length and :math:`S_r^2` the population
  variance (divisor :math:`M`) of the items' exposure rates. It assumes every examinee takes
  :math:`n` items. It is :math:`n/M` when every item is equally exposed and 1 when every examinee
  takes the same test; higher values mean examinees share more items, which weakens test security.

  Parameters
  ----------
  exposure_rates : numpy.ndarray
      Each item's exposure rate, the share of examinees who took it.
  test_size : int
      The number of items in each test.

  Returns
  -------
  float
      The overlap rate, between 0 and 1.

  Raises
  ------
  ValueError
      If an exposure rate is outside [0, 1], if ``test_size`` is not positive, or if it is larger
      than the bank.

  Examples
  --------
  >>> import numpy
  >>> from catsim.metrics import overlap_rate
  >>> overlap_rate(numpy.array([1.0, 0.5, 0.5, 0.0]), test_size=2)
  0.75
  """
  if numpy.any(exposure_rates < 0) or numpy.any(exposure_rates > 1):
    msg = "Exposure rates must be between 0 and 1"
    raise ValueError(msg)

  if test_size <= 0:
    msg = "Test size must be positive"
    raise ValueError(msg)

  bank_size = exposure_rates.shape[0]

  if test_size > bank_size:
    msg = f"Test size ({test_size}) cannot be larger than bank size ({bank_size})"
    raise ValueError(msg)

  var_r = float(numpy.var(exposure_rates))
  return (bank_size / test_size) * var_r + (test_size / bank_size)
