"""Functions used specifically during the application/simulation of computerized adaptive tests."""

import operator
import random

import numpy
from numpy.random import Generator

from catsim import irt


def dodd(theta: float, items: numpy.ndarray, correct: bool) -> float:
  r"""Estimate :math:`\hat{\theta}` when the response vector is composed entirely of 1s or 0s.

  Method proposed by [Dod90]_.

  .. math::

      \hat{\theta}_{t+1} = \left\lbrace \begin{array}{ll}
      \hat{\theta}_t+\frac{b_{max}-\hat{\theta_t}}{2} & \text{if } X_t = 1 \\
        \hat{\theta}_t-\frac{\hat{\theta}_t-b_{min}}{2} & \text{if }  X_t = 0
      \end{array} \right\rbrace

  :param theta: the initial ability level
  :param items: a numpy array containing the parameters of the items in the
                database. This is necessary to capture the maximum and minimum
                difficulty levels necessary for the method.
  :param correct: a boolean value informing if the examinee has answered only correctly
                  (`True`) or incorrectly (`False`) up until now
  :returns: a new estimation for :math:`\theta`
  """
  b = items[:, 1]

  return theta + ((max(b) - theta) / 2) if correct else theta - ((theta - min(b)) / 2)


def bias(
  actual: list[float] | numpy.ndarray,
  predicted: list[float] | numpy.ndarray,
) -> float:
  r"""Compute the test bias, an evaluation criterion for computerized adaptive test methodolgies [Chang2001]_.

  The value is computed as:

  .. math:: Bias = \frac{\sum_{i=1}^{N} (\hat{\theta}_i - \theta_{i})}{N}

  where :math:`\hat{\theta}_i` is examinee :math:`i` estimated ability and
  :math:`\theta_i` is examinee :math:`i` actual ability.

  :param actual: a list or 1-D numpy array containing the true ability values
  :param predicted: a list or 1-D numpy array containing the estimated ability values
  :returns: the bias between the predicted values and actual values.
  """
  if len(actual) != len(predicted):
    msg = "actual and predicted vectors need to be the same size"
    raise ValueError(msg)
  return float(numpy.mean(list(map(operator.sub, predicted, actual))))


def mse(
  actual: list[float] | numpy.ndarray,
  predicted: list[float] | numpy.ndarray,
) -> float:
  r"""Compute the mean squared error (MSE) between two lists or 1-D numpy arrays.

  The MSE isused when measuring the precision with which a computerized adaptive test estimates examinees
  abilities [Chang2001]_.

  The value is computed as:

  .. math:: MSE = \frac{\sum_{i=1}^{N} (\hat{\theta}_i - \theta_{i})^2}{N}

  where :math:`\hat{\theta}_i` is examinee :math:`i` estimated ability and
  :math:`\hat{\theta}_i` is examinee :math:`i` actual ability.

  :param actual: a list or 1-D numpy array containing the true ability values
  :param predicted: a list or 1-D numpy array containing the estimated ability values
  :returns: the mean squared error between the predicted values and actual values.
  """
  if len(actual) != len(predicted):
    msg = "Actual and predicted vectors need to be the same size"
    raise ValueError(msg)
  return float(numpy.mean([x * x for x in list(map(operator.sub, predicted, actual))]))


def rmse(
  actual: list[float] | numpy.ndarray,
  predicted: list[float] | numpy.ndarray,
) -> float:
  r"""Compute the root mean squared error (RMSE) between two lists or 1-D numpy arrays.

  A common value used when measuring the precision with which a computerized adaptive test estimates examinees
  abilities [Bar10]_.

  The value is computed as:

  .. math:: RMSE = \sqrt{\frac{\sum_{i=1}^{N} (\hat{\theta}_i - \theta_{i})^2}{N}}

  where :math:`\hat{\theta}_i` is examinee :math:`i` estimated ability and
  :math:`\hat{\theta}_i` is examinee :math:`i` actual ability.

  :param actual: a list or 1-D numpy array containing the true ability values
  :param predicted: a list or 1-D numpy array containing the estimated ability values
  :returns: the root mean squared error between the predicted values and actual values.
  """
  if len(actual) != len(predicted):
    msg = "actual and predicted vectors need to be the same size"
    raise ValueError(msg)
  return numpy.sqrt(mse(actual, predicted))


def overlap_rate(usages: numpy.ndarray, test_size: int) -> float:
  r"""Compute the test overlap rate.

  An average measure of how much of the test two examinees take is equal [Bar10]_. It is given by:

  .. math:: T=\frac{N}{Q}S_{r}^2 + \frac{Q}{N}

  If, for example :math:`T = 0.5`, it means that the tests of two random examinees have 50% of equal items.

  :param usages: a list or numpy.ndarray containing the number of
                times each item was used in the tests.
  :param test_size: an integer informing the number of items in a test.
  :returns: test overlap rate.
  """
  if any(usages > test_size):
    msg = "There are items that have been used more times than there were tests"
    raise ValueError(msg)

  bank_size = usages.shape[0]
  var_r = numpy.var(usages)

  return (bank_size / test_size) * var_r + (test_size / bank_size)


def generate_item_bank(
  n: int,
  itemtype: irt.NumParams = irt.NumParams.PL4,
  corr: float = 0,
  rng: Generator | None = None,
  seed: int = 0,
) -> numpy.ndarray:
  """Generate a synthetic item bank whose parameters approximately follow real-world parameters.

  As proposed by [Bar10]_, item parameters are extracted from the following probability distributions:

  * discrimination: :math:`N(1.2, 0.25)`

  * difficulty: :math:`N(0,  1)`

  * pseudo-guessing: :math:`N(0.25, 0.02)`

  * upper asymptote: :math:`U(0.94, 1)`

  :param n: how many items are to be generated
  :param itemtype: either ``1PL``, ``2PL``, ``3PL`` or ``4PL`` for the one-, two-,
                   three- or four-parameter logistic model
  :param corr: the correlation between item discrimination and difficulty. If
               ``itemtype == '1PL'``, it is ignored.
  :param rng: Optional random number generator to generate the item bank. If not passed, one will be created.
  :param seed: Seed used to create a random number generator, if one is not provided. Defaults to 0.
  :return: an ``n x 4`` numerical matrix containing item parameters
  :rtype: numpy.ndarray

  >>> generate_item_bank(5, irt.NumParams.PL1)
  >>> generate_item_bank(5, irt.NumParams.PL2)
  >>> generate_item_bank(5, irt.NumParams.PL3)
  >>> generate_item_bank(5, irt.NumParams.PL4)
  >>> generate_item_bank(5, irt.NumParams.PL4, corr=0)
  """
  if not isinstance(itemtype, irt.NumParams):
    msg = "itemtype must be of type irt.NumParams"
    raise TypeError(msg)

  means = [0, 1.2]
  stds = [1, 0.25]
  covs = [
    [stds[0] ** 2, stds[0] * stds[1] * corr],
    [stds[0] * stds[1] * corr, stds[1] ** 2],
  ]

  rng = rng or numpy.random.default_rng(seed)

  b, a = rng.multivariate_normal(means, covs, n).T

  # if by chance there is some discrimination value below zero
  # this makes the problem go away
  if any(disc < 0 for disc in a):
    min_disc = min(a)
    a = [disc + abs(min_disc) for disc in a]

  if itemtype not in {"2PL", "3PL", "4PL"}:
    a = numpy.ones(n)

  c = rng.normal(0.25, 0.02, n).clip(min=0) if itemtype in {"3PL", "4PL"} else numpy.zeros(n)
  d = rng.uniform(0.94, 1, n) if itemtype == "4PL" else numpy.ones(n)

  return irt.normalize_item_bank(numpy.array([a, b, c, d, numpy.zeros(n)]).T)


def random_response_vector(size: int) -> list:
  """Generate a list of random boolean values of the given size.

  Args:
    size (int): The size of the list to be generated.

  Returns:
    list: A list of random boolean values.
  """
  return [bool(random.getrandbits(1)) for _ in range(size)]


if __name__ == "__main__":
  import doctest

  doctest.testmod()
