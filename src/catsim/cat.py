"""Functions used specifically during the application/simulation of computerized adaptive tests."""

import random

import numpy
import numpy.typing as npt

from catsim.item_bank import ItemBank


def dodd(theta: float, item_bank: ItemBank, correct: bool) -> float:
  r"""Estimate :math:`\hat{\theta}` when the response vector is composed entirely of 1s or 0s.

  Method proposed by [Dod90]_. This heuristic prevents the maximum likelihood
  estimator from returning infinity when all responses are correct or negative
  infinity when all responses are incorrect.

  .. math::

      \hat{\theta}_{t+1} = \left\lbrace \begin{array}{ll}
      \hat{\theta}_t+\frac{b_{max}-\hat{\theta_t}}{2} & \text{if } X_t = 1 \\
        \hat{\theta}_t-\frac{\hat{\theta}_t-b_{min}}{2} & \text{if }  X_t = 0
      \end{array} \right\rbrace

  Parameters
  ----------
  theta : float
      The initial ability level estimate.
  item_bank : ItemBank
      An ItemBank containing all items in the bank.
      This is necessary to capture the maximum and minimum difficulty levels
      required by the method.
  correct : bool
      Boolean value indicating if the examinee has answered only correctly
      (True) or only incorrectly (False) up until now.

  Returns
  -------
  float
      A new estimation for :math:`\theta` that avoids infinite values.
  """
  b = item_bank.difficulty

  return theta + ((max(b) - theta) / 2) if correct else theta - ((theta - min(b)) / 2)


def bias(
  actual: npt.ArrayLike,
  predicted: npt.ArrayLike,
) -> float:
  r"""Compute the test bias, an evaluation criterion for computerized adaptive test methodologies [Chang2001]_.

  The value is computed as:

  .. math:: Bias = \frac{\sum_{i=1}^{N} (\hat{\theta}_i - \theta_{i})}{N}

  where :math:`\hat{\theta}_i` is examinee :math:`i` estimated ability and
  :math:`\theta_i` is examinee :math:`i` actual ability. A positive bias indicates
  systematic overestimation, while a negative bias indicates systematic underestimation.

  Parameters
  ----------
  actual : npt.ArrayLike
      Array-like (list, tuple, or numpy array) containing the true ability values (float type).
  predicted : npt.ArrayLike
      Array-like (list, tuple, or numpy array) containing the estimated ability values (float type).

  Returns
  -------
  float
      The bias between the predicted values and actual values.

  Raises
  ------
  ValueError
      If actual and predicted vectors are not the same size.
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
  r"""Compute the mean squared error (MSE) between two array-like objects.

  The MSE is used when measuring the precision with which a computerized adaptive
  test estimates examinees abilities [Chang2001]_. Lower MSE values indicate better
  estimation accuracy.

  The value is computed as:

  .. math:: MSE = \frac{\sum_{i=1}^{N} (\hat{\theta}_i - \theta_{i})^2}{N}

  where :math:`\hat{\theta}_i` is examinee :math:`i` estimated ability and
  :math:`\theta_i` is examinee :math:`i` actual ability.

  Parameters
  ----------
  actual : npt.ArrayLike
      Array-like (list, tuple, or numpy array) containing the true ability values (float type).
  predicted : npt.ArrayLike
      Array-like (list, tuple, or numpy array) containing the estimated ability values (float type).

  Returns
  -------
  float
      The mean squared error between the predicted values and actual values.

  Raises
  ------
  ValueError
      If actual and predicted vectors are not the same size.
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
  r"""Compute the root mean squared error (RMSE) between two array-like objects.

  A common value used when measuring the precision with which a computerized adaptive
  test estimates examinees abilities [Bar10]_. RMSE is in the same units as the ability
  scale, making it easier to interpret than MSE.

  The value is computed as:

  .. math:: RMSE = \sqrt{\frac{\sum_{i=1}^{N} (\hat{\theta}_i - \theta_{i})^2}{N}}

  where :math:`\hat{\theta}_i` is examinee :math:`i` estimated ability and
  :math:`\theta_i` is examinee :math:`i` actual ability.

  Parameters
  ----------
  actual : npt.ArrayLike
      Array-like (list, tuple, or numpy array) containing the true ability values (float type).
  predicted : npt.ArrayLike
      Array-like (list, tuple, or numpy array) containing the estimated ability values (float type).

  Returns
  -------
  float
      The root mean squared error between the predicted values and actual values.

  Raises
  ------
  ValueError
      If actual and predicted vectors are not the same size.
  """
  return numpy.sqrt(mse(actual, predicted))


def overlap_rate(exposure_rates: npt.NDArray[numpy.floating], test_size: int) -> float:
  r"""Compute the test overlap rate.

  An average measure of how much of the test two examinees take is equal [Bar10]_.
  The overlap rate provides insight into test security: higher values indicate that
  examinees see more similar items, potentially compromising test security.

  The overlap rate is given by:

  .. math:: T=\frac{N}{Q}S_{r}^2 + \frac{Q}{N}

  where :math:`N` is the bank size, :math:`Q` is the test size, and :math:`S_{r}^2`
  is the variance of exposure rates.

  If, for example :math:`T = 0.5`, it means that the tests of two random examinees
  have 50% of equal items.

  Parameters
  ----------
  exposure_rates : numpy.ndarray
      A numpy array containing the exposure rate (proportion) of each item, where
      each value is between 0 and 1, representing the proportion of examinees who
      received that item.
  test_size : int
      The number of items in a test.

  Returns
  -------
  float
      Test overlap rate between 0 and 1.

  Raises
  ------
  ValueError
      If exposure rates are not between 0 and 1, if test size is not positive,
      or if test size is larger than bank size.
  """
  # Validate that exposure_rates contains proportions
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

  # Compute variance of exposure rates
  var_r = numpy.var(exposure_rates)

  # Apply Barrada et al. (2010) formula
  return (bank_size / test_size) * var_r + (test_size / bank_size)


def random_response_vector(size: int) -> list[bool]:
  """Generate a list of random boolean values of the given size.

  This function is useful for testing and simulation purposes when you need
  a random response pattern.

  Parameters
  ----------
  size : int
      The size of the list to be generated. Must be non-negative.

  Returns
  -------
  list[bool]
      A list of random boolean values with equal probability of True and False.
  """
  return [bool(random.getrandbits(1)) for _ in range(size)]


if __name__ == "__main__":
  import doctest

  doctest.testmod()
