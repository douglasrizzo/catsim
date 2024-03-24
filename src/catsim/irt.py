import math
from enum import Enum
from math import pi  # noqa: F401
from typing import Any

import numexpr
import numpy


class NumParams(Enum):
  """Enumerator that informs how many parameters each logistic model of IRT has.

  Created to avoid accessing magic numbers in the code.
  """

  PL1 = 1
  PL2 = 2
  PL3 = 3
  PL4 = 4


def icc(theta: float, a: float, b: float, c: float = 0, d: float = 1) -> float:
  """Compute the Item Response Theory four-parameter logistic function.

  Args:
    theta (float): the individual's ability value.
    a (float): the discrimination parameter of the item.
    b (float): the item difficulty parameter.
    c (float, optional): the item pseudo-guessing parameter.
    d (float, optional): the item upper asymptote.

  Returns:
    float: the probability of the individual responding correctly to the item.
  """
  return c + ((d - c) / (1 + math.e ** (-a * (theta - b))))


def _split_params(items: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
  """Split the item matrix parameters into columns.

  Args:
    items (np.ndarray): An item matrix with four columns representing four parameters.

  Returns:
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A 4-tuple with each column in a different slot.
  """
  return items[:, 0], items[:, 1], items[:, 2], items[:, 3]


def detect_model(items: numpy.ndarray) -> int:
  """Detects which logistic model an item matrix fits into.

  :param items: an item matrix
  :return: an int between 1 and 4 denoting the logistic model of the given item matrix
  """
  a, _, c, d = _split_params(items)

  if any(d != 1):
    return 4
  if any(c != 0):
    return 3
  if len(set(a)) > 1:
    return 2
  return 1


def icc_hpc(theta: float, items: numpy.ndarray) -> numpy.ndarray:  # noqa: ARG001
  """Compute item characteristic functions for all items in a numpy array at once.

  Args:
      theta: the individual's ability value.
      items: array containing the four item parameters.

  Returns:
      ndarray: an array of all item characteristic functions, given the current theta
  """
  _a, _b, _c, _d = _split_params(items)
  return numexpr.evaluate("_c + ((_d - _c) / (1 + exp((-_a * (theta - _b)))))")


def inf_hpc(theta: float, items: numpy.ndarray) -> numpy.ndarray:
  """Compute the information values for all items in a numpy array using numpy and numexpr.

  Args:
      theta (float): The individual's ability value.
      items (numpy.ndarray): Array containing the four item parameters.

  Returns:
      numpy.ndarray: An array of all item information values, given the current theta.
  """
  _a, _b, _c, _d = _split_params(items)
  _p = icc_hpc(theta, items)

  return numexpr.evaluate("(_a ** 2 * (_p - _c) ** 2 * (_d - _p) ** 2) / ((_d - _c) ** 2 * _p * (1 - _p))")


def inf(theta: float, a: float, b: float, c: float = 0, d: float = 1) -> float:
  r"""Compute the information value of an item using the Item Response Theory four-parameter logistic model function.

  References are given in [Ayala2009]_, [Magis13]_.

  Args:
    theta: the individual's ability value. This parameter value has
          no boundary, but if a distribution of the form
          :math:`N(0, 1)` was used to estimate the parameters, then
          :math:`-4 \leq \theta \leq 4`.
    a: the discrimination parameter of the item, usually a positive
      value in which :math:`0.8 \leq a \leq 2.5`.
    b: the item difficulty parameter. This parameter value has no
      boundary, but if a distribution of the form :math:`N(0, 1)` was
      used to estimate the parameters, then :math:`-4 \leq b \leq 4`.
    c: the item pseudo-guessing parameter. Being a probability,
      :math:`0 \leq c \leq 1`, but items considered good usually have
      :math:`c \leq 0.2`.
    d: the item upper asymptote. Being a probability,
      :math:`0 \leq d \leq 1`, but items considered good usually have
      :math:`d \approx 1`.

  Returns:
    The information value of the item at the designated `theta` point.
  """
  p = icc(theta, a, b, c, d)

  return (a**2 * (p - c) ** 2 * (d - p) ** 2) / ((d - c) ** 2 * p * (1 - p))


def test_info(theta: float, items: numpy.ndarray) -> float:
  r"""Compute the test information of a test at a specific :math:`\theta` value.

  .. math:: I(\theta) = \sum_{j \in J} I_j(\theta)

  Args:
    theta: An ability value.
    items: A matrix containing item parameters.

  Returns:
    The test information at `theta` for a test represented by `items`.
  """
  return float(numpy.sum(inf_hpc(theta, items)))


def var(theta: float, items: numpy.ndarray) -> float:
  r"""Compute the variance (:math:`Var`) of the ability estimate of a test at a specific :math:`\theta` value.

  Reference is given in [Ayala2009]_.

  Args:
      theta (float): An ability value.
      items (numpy.ndarray): A matrix containing item parameters.

  Returns:
      float: The variance of ability estimation at `theta` for a test represented by `items`.

  Raises:
      ZeroDivisionError: If the test information is zero, returns negative infinity.
  """
  try:
    return 1 / test_info(theta, items)
  except ZeroDivisionError:
    return float("-inf")


def see(theta: float, items: numpy.ndarray) -> float:
  r"""Compute the standard error of estimation (:math:`SEE`) of a test at a specific :math:`\\theta` value [Ayala2009]_.

  Args:
      theta (float): An ability value.
      items (numpy.ndarray): A matrix containing item parameters.

  Returns:
      float: The standard error of estimation at `theta` for a test represented by `items`.
  """
  try:
    return math.sqrt(var(theta, items))
  except ValueError:
    return float("inf")


def reliability(theta: float, items: numpy.ndarray) -> float:
  r"""Compute test reliability [Thissen00]_.

  .. math:: Rel = 1 - \\frac{1}{I(\\theta)}

  Test reliability is a measure of internal consistency for the test, similar to Cronbach's :math:`\\alpha` in Classical
  Test Theory. Its value is always lower than 1, with values close to 1 indicating good reliability. If
  :math:`I(\\theta) < 1`, :math:`Rel < 0` and in these cases it does not make sense, but usually the application of
  additional items solves this problem.

  Args:
    theta: An ability value.
    items: A matrix containing item parameters.

  Returns:
    float: The test reliability at `theta` for a test represented by `items`.
  """
  return 1 - var(theta, items)


def max_info(a: float = 1, b: float = 0, c: float = 0, d: float = 1) -> float:
  r"""Return the :math:`\theta` value to which the item with the given parameters gives maximum information.

  For the 1-parameter and 2-parameter logistic models, this :math:`\theta` corresponds to where :math:`b = 0.5`. In the
  3-parameter and 4-parameter logistic models, however, this value is given by ([Magis13]_)

  .. math:: argmax_{\theta}I(\theta) = b + \frac{1}{a} log \left(\frac{x^* - c}{d - x^*}\right)

  where

  .. math::

    x^* = 2 \sqrt{\frac{-u}{3}} cos\left\{\frac{1}{3}acos\left(-\frac{v}{2}\sqrt{\frac{27}{-u^3}}\right)+
    \frac{4 \pi}{3}\right\} + 0.5

  .. math:: u = -\frac{3}{4} + \frac{c + d - 2cd}{2}

  .. math:: v = -\frac{c + d - 1}{4}

  A few results can be seen in the plots below:

  .. plot::

      from catsim.cat import generate_item_bank
      from catsim import plot
      items = generate_item_bank(2)
      for item in items:
          plot.item_curve(item[0], item[1], item[2], item[3], ptype='iic', max_info=True)

  Args:
    a: item discrimination parameter
    b: item difficulty parameter
    c: item pseudo-guessing parameter
    d: item upper asymptote

  Returns:
    The theta value to which the item with the given parameters gives maximum information
  """
  # for explanations on finding the following values, see referenced work in function description
  if d == 1:
    if c == 0:
      return b
    return b + (1 / a) * math.log((1 + math.sqrt(1 + 8 * c)) / 2)

  u = -(3 / 4) + ((c + d - 2 * c * d) / 2)
  v = (c + d - 1) / 4
  x_star = (
    2
    * math.sqrt(-u / 3)
    * math.cos((1 / 3) * math.acos(-(v / 2) * math.sqrt(27 / (-math.pow(u, 3)))) + (4 * math.pi / 3))
    + 0.5
  )
  return b + (1 / a) * math.log((x_star - c) / (d - x_star))


def max_info_hpc(items: numpy.ndarray) -> numpy.ndarray:
  """Parallelized version of :py:func:`max_info` using :py:mod:`numpy` and :py:mod:`numexpr`.

  Args:
    items (numpy.ndarray): Array containing the four item parameters.

  Returns:
    numpy.ndarray: An array of all theta values that maximize the information function of each item.
  """
  _a, _b, _c, _d = _split_params(items)

  if all(_d == 1):
    if all(_c == 0):
      return _b
    return numexpr.evaluate("_b + (1 / _a) * log((1 + sqrt(1 + 8 * _c)) / 2)")

  _u = numexpr.evaluate("-(3 / 4) + ((_c + _d - 2 * _c * _d) / 2)")
  _v = numexpr.evaluate("(_c + _d - 1) / 4")
  _x_star = numexpr.evaluate(
    "2 * sqrt(-_u / 3) * cos((1 / 3) * arccos(-(_v / 2) * sqrt(27 / -(_u ** 3))) + (4 * pi / 3)) + 0.5"
  )

  return numexpr.evaluate("_b + (1 / _a) * log((_x_star - _c) / (_d - _x_star))")


def log_likelihood(est_theta: float, response_vector: list[bool], administered_items: numpy.ndarray) -> float:
  r"""Compute the log-likelihood of an ability, given a response vector and the parameters of the answered items.

  Reference is given in [Ayala2009]_.

  The likelihood function of a given :math:`\theta` value given the answers to :math:`I` items is given by:

  .. math:: L(X_{Ij} | \theta_j, a_I, b_I, c_I, d_I) = \prod_{i=1} ^ I P_{ij}(\theta)^{X_{ij}} Q_{ij}(\theta)^{1-X_{ij}}

  Finding the maximum of :math:`L(X_{Ij})` includes using the product rule of derivations.
  Since :math:`L(X_{Ij})` has :math:`j` parts, it can be quite complicated to do so. Also,
  for computational reasons, the product of probabilities can quickly tend to 0, so it is
  common to use the log-likelihood in maximization/minimization problems, transforming the
  product of probabilities in a sum of probabilities:

   .. math:: \log L(X_{Ij} | \theta_j, a_I, b_I, c_I, d_I) = \sum_{i=1} ^ I
             \left\lbrace x_{ij} \log P_{ij}(\theta)+ (1 - x_{ij}) \log
             Q_{ij}(\theta) \right\rbrace

  Args:
    est_theta (float): Estimated ability value.
    response_vector (List[bool]): A list containing the response vector.
    administered_items (np.ndarray): An array containing the parameters of the answered items.

  Returns:
    float: Log-likelihood of a given ability value, given the responses to the administered items.

  Raises:
    ValueError: If the response vector and administered items do not have the same number of items, or if the response
    vector contains elements other than True or False.
  """
  if len(response_vector) != administered_items.shape[0]:
    msg = "Response vector and administered items must have the same number of items"
    raise ValueError(msg)
  if len(set(response_vector) - {True, False}) > 0:
    msg = "Response vector must contain only Boolean elements"
    raise ValueError(msg)
  _ps = icc_hpc(est_theta, administered_items)
  return numexpr.evaluate("sum(where(response_vector, log(_ps), log(1 - _ps)))")


def negative_log_likelihood(est_theta: float, *args: tuple[Any, ...]) -> float:
  """Compute the negative log-likelihood of a ability value, given an array of item parameters and their answers.

  This function is used by the functions in :py:mod:`scipy.optimize` that search for minima instead of maxima. Its
  output is simply the value of :math:`-` :py:func:`log_likelihood`.

  Args:
    est_theta (float): estimated ability value
    *args:
      - response_vector (List[bool]): a Boolean list containing the response vector
      - administered_items (np.ndarray): a numpy array containing the parameters of the answered items

  Returns:
    float: negative log-likelihood of a given ability value, given the responses to the administered items
  """
  return -log_likelihood(est_theta, args[0], args[1])


def normalize_item_bank(items: numpy.ndarray) -> numpy.ndarray:
  r"""Normalize an item matrix so that it conforms to the standard used by catsim.

  The item matrix must have dimension :math:`n \times 4`, in which column 1 represents item
  discrimination, column 2 represents item difficulty, column 3 represents the pseudo-guessing
  parameter and column 4 represents the item upper asymptote.

  If the matrix has one column, it is assumed to be the difficulty column and the other
  three columns are added such that items simulate the 1-parameter logistic model.

  If the matrix has two columns, they are assumed to be the discrimination and difficulty
  columns, respectively. The pseudo-guessing and upper asymptote columns are added such that
  items simulate the 2-parameter logistic model.

  If the matrix has three columns, they are assumed to be the discrimination, difficulty
  and pseudo-guessing columns, respectively. The upper asymptote column is added such that
  items simulate the 3-parameter logistic model.

  Args:
    items (numpy.ndarray): the item matrix.

  Returns:
    numpy.ndarray: an :math:`n \times 4` item matrix conforming to the 4 parameter logistic model.
  """
  if len(items.shape) == 1:
    items = numpy.expand_dims(items, axis=0)
  if items.shape[1] == NumParams.PL1.value:
    items = numpy.append(numpy.ones((items.shape[0], 1)), items, axis=1)
  if items.shape[1] == NumParams.PL2.value:
    items = numpy.append(items, numpy.zeros((items.shape[0], 1)), axis=1)
  if items.shape[1] == NumParams.PL3.value:
    items = numpy.append(items, numpy.ones((items.shape[0], 1)), axis=1)

  return items


def validate_item_bank(items: numpy.ndarray, raise_err: bool = False) -> None:
  r"""Validate the shape and parameters in the item matrix so that it conforms to the standard used by catsim.

  The item matrix must have dimension nx4, in which column 1 represents item discrimination, column 2 represents item
  difficulty, column 3 represents the pseudo-guessing parameter and column 4 represents the item upper asymptote.

  The item matrix must have at least one line, representing an item, and exactly four columns, representing the
  four discrimination, difficulty, pseudo-guessing and upper asymptote parameters (:math:`a`, :math:`b`, :math:`c` and
  :math:`d`), respectively. The item matrix is considered valid if, for all items in the matrix,
  :math:`a > 0 \wedge 0 < c < 1 \wedge 0 < d < 1`.

  Args:
    items (numpy.ndarray): the item matrix.
    raise_err (bool): whether to raise a ValueError if validation fails.

  Raises:
    TypeError: if the item matrix is not of type numpy.ndarray or does not meet catsim standards.
  """
  if not isinstance(items, numpy.ndarray):
    msg = "Item matrix is not of type numpy.ndarray"
    raise TypeError(msg)

  err = ""

  if len(items.shape) == 1:
    err += "Item matrix has only one dimension."
  elif items.shape[1] > NumParams.PL4.value:
    print(
      "\nItem matrix has more than 4 columns. catsim tends to add \
            columns to the matrix during the simulation, so it's not a good idea to keep them."
    )
  elif items.shape[1] < NumParams.PL4.value:
    if items.shape[1] == NumParams.PL1.value:
      err += "\nItem matrix has no discrimination, pseudo-guessing or upper asymptote parameter columns"
    elif items.shape[1] == NumParams.PL2.value:
      err += "\nItem matrix has no pseudo-guessing or upper asymptote parameter columns"
    elif items.shape[1] == NumParams.PL3.value:
      err += "\nItem matrix has no upper asymptote parameter column"
  else:
    if any(items[:, 0] < 0):
      err += "\nThere are items with discrimination < 0"
    if any(items[:, 2] < 0):
      err += "\nThere are items with pseudo-guessing < 0"
    if any(items[:, 2] > 1):
      err += "\nThere are items with pseudo-guessing > 1"
    if any(items[:, 3] > 1):
      err += "\nThere are items with upper asymptote > 1"
    if any(items[:, 3] < 0):
      err += "\nThere are items with upper asymptote < 0"

  if len(err) > 0 and raise_err:
    raise ValueError(err)

  print(err)
