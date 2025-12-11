"""Item Response Theory (IRT) functions and utilities for computerized adaptive testing.

This module provides core IRT functions including item characteristic curves (ICC),
item information functions, and various utility functions for parameter estimation
and scale transformations.
"""

import math
from enum import Enum
from typing import Any

import numexpr
import numpy
import numpy.typing as npt
from scipy import stats

THETA_MIN_TYPICAL = -4.0
"""Typical lower bound for ability (theta) estimation.

This value covers approximately 99.99% of the population (±4 standard deviations)
assuming abilities are normally distributed N(0, 1). Commonly used in IRT literature
for scale transformations and reporting."""

THETA_MAX_TYPICAL = 4.0
"""Typical upper bound for ability (theta) estimation.

This value covers approximately 99.99% of the population (±4 standard deviations)
assuming abilities are normally distributed N(0, 1). Commonly used in IRT literature
for scale transformations and reporting."""

THETA_MIN_EXTENDED = -6.0
"""Extended lower bound for ability (theta) estimation.

This value covers >99.9999% of the population (±6 standard deviations) assuming
abilities are normally distributed N(0, 1). Recommended for numerical search
algorithms to avoid ceiling/floor effects during estimation. Using bounds wider
than the item bank difficulty range prevents artificial restrictions on ability
estimates."""

THETA_MAX_EXTENDED = 6.0
"""Extended upper bound for ability (theta) estimation.

This value covers >99.9999% of the population (±6 standard deviations) assuming
abilities are normally distributed N(0, 1). Recommended for numerical search
algorithms to avoid ceiling/floor effects during estimation. Using bounds wider
than the item bank difficulty range prevents artificial restrictions on ability
estimates."""


class NumParams(Enum):
  """Enumerator that informs how many parameters each logistic model of IRT has.

  Created to avoid accessing magic numbers in the code. Each value corresponds
  to the number of parameters in the respective IRT model.

  Attributes
  ----------
  PL1 : int
      One-parameter logistic model (Rasch model).
  PL2 : int
      Two-parameter logistic model.
  PL3 : int
      Three-parameter logistic model.
  PL4 : int
      Four-parameter logistic model.
  """

  PL1 = 1
  PL2 = 2
  PL3 = 3
  PL4 = 4


def icc(theta: float, a: float, b: float, c: float = 0, d: float = 1) -> float:
  r"""Compute the Item Response Theory four-parameter logistic function [Magis13]_.

  The item characteristic curve (ICC) represents the probability that an examinee
  with ability :math:`\theta` will correctly answer an item with the given parameters.

  .. math:: P(X_i = 1| \theta) = c_i + \frac{d_i-c_i}{1+ e^{-a_i(\theta-b_i)}}

  Parameters
  ----------
  theta : float
      The individual's ability value. This parameter value has no boundary, but
      if a distribution of the form :math:`N(0, 1)` was used to estimate the
      parameters, then typically :math:`-4 \leq \theta \leq 4`.
  a : float
      The discrimination parameter of the item, usually a positive value in which
      :math:`0.8 \leq a \leq 2.5`. Higher values indicate better discrimination
      between ability levels.
  b : float
      The item difficulty parameter. This parameter value has no boundaries, but
      it must be in the same value space as `theta` (usually :math:`-4 \leq b \leq 4`).
      Higher values indicate more difficult items.
  c : float, optional
      The item pseudo-guessing parameter. Being a probability, :math:`0\leq c \leq 1`,
      but items considered good usually have :math:`c \leq 0.2`. Represents the
      lower asymptote (probability of guessing correctly). Default is 0.
  d : float, optional
      The item upper asymptote. Being a probability, :math:`0\leq d \leq 1`, but
      items considered good usually have :math:`d \approx 1`. Represents the maximum
      probability of correct response. Default is 1.

  Returns
  -------
  float
      The probability :math:`P(X_i = 1| \theta)` of a correct response, a value
      between c and d.
  """
  return c + ((d - c) / (1 + math.e ** (-a * (theta - b))))


def _split_params(
  items: npt.NDArray[numpy.floating[Any]],
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
  """Split the item matrix parameters into columns.

  This is a helper function to extract item parameters for vectorized operations.

  Parameters
  ----------
  items : numpy.ndarray
      An item matrix with at least four columns representing the four IRT parameters.

  Returns
  -------
  tuple of numpy.ndarray
      A 4-tuple containing (a, b, c, d) where:
      - a: discrimination parameters
      - b: difficulty parameters
      - c: pseudo-guessing parameters
      - d: upper asymptote parameters
  """
  return items[:, 0], items[:, 1], items[:, 2], items[:, 3]


def detect_model(items: npt.NDArray[numpy.floating[Any]]) -> int:
  """Detect which logistic model an item matrix fits into.

  Examines the item parameters to determine if they represent a 1PL, 2PL, 3PL,
  or 4PL model based on which parameters vary.

  Parameters
  ----------
  items : numpy.ndarray
      An item matrix with at least four columns.

  Returns
  -------
  int
      An integer between 1 and 4 denoting the logistic model:
      - 1: Rasch/1PL (all discrimination equal, no guessing, no upper asymptote)
      - 2: 2PL (varying discrimination, no guessing, no upper asymptote)
      - 3: 3PL (varying discrimination, guessing present, no upper asymptote variation)
      - 4: 4PL (all parameters vary)
  """
  a, _, c, d = _split_params(items)

  if any(d != 1):
    return 4
  if any(c != 0):
    return 3
  if len(set(a)) > 1:
    return 2
  return 1


def icc_hpc(
  theta: float | npt.NDArray[numpy.floating[Any]], items: npt.NDArray[numpy.floating[Any]]
) -> npt.NDArray[numpy.floating[Any]]:
  """Compute item characteristic functions for all items in a numpy array at once.

  This is a high-performance computing (HPC) version that uses vectorized operations
  with numpy and numexpr for efficient batch processing of multiple items.

  Parameters
  ----------
  theta : float or numpy.ndarray
      The individual's ability value(s). Can be a scalar or an array for element-wise
      computation with items (via numpy broadcasting).
  items : numpy.ndarray
      Array containing item parameters with at least four columns
      [a, b, c, d, ...].

  Returns
  -------
  numpy.ndarray
      An array of probabilities, one for each item, representing the probability
      of a correct response given the current theta.
  """
  a, b, c, d = _split_params(items)
  return numexpr.evaluate(
    "c + ((d - c) / (1 + exp((-a * (theta - b)))))",
    local_dict={"a": a, "b": b, "c": c, "d": d, "theta": theta},
  )


def inf_hpc(
  theta: float | npt.NDArray[numpy.floating[Any]], items: npt.NDArray[numpy.floating[Any]]
) -> npt.NDArray[numpy.floating[Any]]:
  """Compute the information values for all items in a numpy array using numpy and numexpr.

  This is a high-performance computing (HPC) version that uses vectorized operations
  for efficient batch processing of item information values.

  Parameters
  ----------
  theta : float or numpy.ndarray
      The individual's ability value(s). Can be a scalar or an array for element-wise
      computation with items (via numpy broadcasting).
  items : numpy.ndarray
      Array containing item parameters with at least four columns [a, b, c, d, ...].

  Returns
  -------
  numpy.ndarray
      Array containing the information values for each item at the given theta.
      Information quantifies how much an item contributes to ability estimation
      precision at the specified ability level.
  """
  a, _b, c, d = _split_params(items)
  p = icc_hpc(theta, items)

  return numexpr.evaluate(
    "(a ** 2 * (p - c) ** 2 * (d - p) ** 2) / ((d - c) ** 2 * p * (1 - p))",
    local_dict={"a": a, "c": c, "d": d, "p": p},
  )


def inf(theta: float, a: float, b: float, c: float = 0, d: float = 1) -> float:
  r"""Compute the information value of an item using the Item Response Theory four-parameter logistic model.

  Item information quantifies how precisely an item can estimate ability at a given
  :math:`\theta` level. Higher information indicates better precision. References
  are given in [Ayala2009]_, [Magis13]_.

  .. math:: I_i(\theta) = \frac{a^2[(P(\theta)-c)]^2[d - P(\theta)]^2}{(d-c)^2[1-P(\theta)]P(\theta)}

  Parameters
  ----------
  theta : float
      The individual's ability value. This parameter value has no boundary, but
      if a distribution of the form :math:`N(0, 1)` was used to estimate the
      parameters, then typically :math:`-4 \leq \theta \leq 4`.
  a : float
      The discrimination parameter of the item, usually a positive value in which
      :math:`0.8 \leq a \leq 2.5`.
  b : float
      The item difficulty parameter. This parameter value has no boundary, but
      if a distribution of the form :math:`N(0, 1)` was used to estimate the
      parameters, then typically :math:`-4 \leq b \leq 4`.
  c : float, optional
      The item pseudo-guessing parameter. Being a probability, :math:`0\leq c \leq 1`,
      but items considered good usually have :math:`c \leq 0.2`. Default is 0.
  d : float, optional
      The item upper asymptote. Being a probability, :math:`0\leq d \leq 1`, but
      items considered good usually have :math:`d \approx 1`. Default is 1.

  Returns
  -------
  float
      The information value of the item at the designated `theta` point.
  """
  p = icc(theta, a, b, c, d)

  return (a**2 * (p - c) ** 2 * (d - p) ** 2) / ((d - c) ** 2 * p * (1 - p))


def test_info(theta: float, items: npt.NDArray[numpy.floating[Any]]) -> float:
  r"""Compute the test information of a test at a specific :math:`\theta` value [Ayala2009]_.

  Test information is the sum of individual item information values and indicates
  the precision of ability estimation at a given ability level.

  .. math:: I(\theta) = \sum_{j \in J} I_j(\theta)

  Parameters
  ----------
  theta : float
      An ability value at which to compute test information.
  items : numpy.ndarray
      A matrix containing item parameters with at least four columns [a, b, c, d, ...].

  Returns
  -------
  float
      The test information at `theta` for a test represented by `items`.
      Higher values indicate more precise ability estimation.
  """
  return float(numpy.sum(inf_hpc(theta, items)))


def var(
  theta: float | None = None,
  items: npt.NDArray[numpy.floating[Any]] | None = None,
  test_inf: float | None = None,
) -> float:
  r"""Compute the variance (:math:`Var`) of the ability estimate of a test at a specific :math:`\theta` value.

  Variance quantifies the uncertainty in the ability estimate. Lower variance
  indicates more precise estimation.

  .. math:: Var = \frac{1}{I(\theta)}

  where :math:`I(\theta)` is the test information (see :py:func:`test_info`).

  Parameters
  ----------
  theta : float or None, optional
      An ability value (required if `test_inf` is not provided). Default is None.
  items : numpy.ndarray or None, optional
      A matrix containing item parameters (required if `test_inf` is not provided).
      Default is None.
  test_inf : float or None, optional
      The test information value. If provided, `theta` and `items` are not required.
      Default is None.

  Returns
  -------
  float
      The variance of ability estimation. Returns negative infinity if test
      information is 0.

  Raises
  ------
  ValueError
      If neither `test_inf` nor both `theta` and `items` are provided.
  """
  if test_inf is None:
    if theta is None or items is None:
      msg = "Either theta and items or test_inf must be passed"
      raise ValueError(msg)
    test_inf = test_info(theta, items)
  if test_inf == 0:
    return float("-inf")
  return 1 / test_inf


def see(theta: float, items: npt.NDArray[numpy.floating[Any]]) -> float:
  r"""Compute the standard error of estimation (:math:`SEE`) of a test at a specific :math:`\theta` value [Ayala2009]_.

  The standard error of estimation is the square root of variance and represents
  the typical error in ability estimation. It is in the same units as the ability
  scale, making it more interpretable than variance.

  .. math:: SEE = \sqrt{\frac{1}{I(\theta)}}

  where :math:`I(\theta)` is the test information (see :py:func:`test_info`).

  Parameters
  ----------
  theta : float
      An ability value.
  items : npt.NDArray[numpy.floating[Any]]
      A matrix containing item parameters.

  Returns
  -------
  float
      The standard error of estimation at `theta` for a test represented by `items`.
      Returns infinity if test information is 0.
  """
  try:
    return math.sqrt(var(theta, items))
  except ValueError:
    return float("inf")


def confidence_interval(
  theta: float, items: npt.NDArray[numpy.floating[Any]], confidence: float = 0.95
) -> tuple[float, float]:
  r"""Compute the confidence interval for an ability estimate.

  The confidence interval is computed using the normal approximation:

  .. math:: CI = \hat{\theta} \pm z_{\alpha/2} \times SEE(\hat{\theta})

  where :math:`z_{\alpha/2}` is the critical value from the standard normal distribution
  corresponding to the desired confidence level, and :math:`SEE` is the standard error
  of estimation.

  Parameters
  ----------
  theta : float
      The estimated ability value.
  items : npt.NDArray[numpy.floating[Any]]
      A matrix containing item parameters for administered items.
  confidence : float, optional
      The confidence level, must be between 0 and 1. Default is 0.95 (95% confidence).
      Common values are 0.90 (90%), 0.95 (95%), and 0.99 (99%).

  Returns
  -------
  tuple[float, float]
      A tuple containing (lower_bound, upper_bound) of the confidence interval.

  Raises
  ------
  ValueError
      If confidence is not between 0 and 1.

  Examples
  --------
  >>> import numpy as np
  >>> items = np.array([[1.0, 0.0, 0.0, 1.0], [1.2, -0.5, 0.0, 1.0]])
  >>> theta = 0.5
  >>> lower, upper = confidence_interval(theta, items, confidence=0.95)
  >>> print(f"95% CI: [{lower:.3f}, {upper:.3f}]")  # doctest: +SKIP
  95% CI: [-0.234, 1.234]
  """
  if not 0 < confidence < 1:
    msg = f"Confidence level must be between 0 and 1, got {confidence}"
    raise ValueError(msg)

  # Compute standard error of estimation
  standard_error = see(theta, items)

  # If SEE is infinite, return infinite bounds
  if math.isinf(standard_error):
    return (float("-inf"), float("inf"))

  # Compute z-score for the given confidence level
  # For a two-tailed test, we need the (1 + confidence) / 2 quantile
  # Common z-scores: 90% = 1.645, 95% = 1.96, 99% = 2.576
  z_score = float(stats.norm.ppf((1 + confidence) / 2))

  # Compute confidence interval bounds
  margin = z_score * standard_error
  lower_bound = theta - margin
  upper_bound = theta + margin

  return (lower_bound, upper_bound)


def theta_to_scale(
  theta: float | npt.NDArray[numpy.floating[Any]],
  scale_min: float = 0,
  scale_max: float = 100,
  theta_min: float = -4,
  theta_max: float = 4,
) -> float | npt.NDArray[numpy.floating[Any]]:
  r"""Convert theta values to a custom score scale using linear transformation.

  This function transforms ability estimates from the standard IRT theta scale
  (typically centered at 0) to any desired score scale (e.g., 0-100, 200-800).

  The linear transformation is:

  .. math:: \text{score} = a \cdot \theta + b

  where :math:`a` and :math:`b` are determined by mapping the theta range to the score range.

  Parameters
  ----------
  theta : float or numpy.ndarray
      Ability value(s) on the theta scale.
  scale_min : float, optional
      Minimum value of the target score scale. Default is 0.
  scale_max : float, optional
      Maximum value of the target score scale. Default is 100.
  theta_min : float, optional
      Minimum theta value to map. Default is -4 (typical lower bound).
  theta_max : float, optional
      Maximum theta value to map. Default is 4 (typical upper bound).

  Returns
  -------
  float or numpy.ndarray
      The transformed score(s) on the target scale. Returns the same type as input.

  Examples
  --------
  >>> # Convert theta to 0-100 scale
  >>> theta_to_scale(0.0)  # Average ability
  50.0
  >>> theta_to_scale(2.0)  # High ability
  75.0
  >>> theta_to_scale(-2.0)  # Low ability
  25.0

  >>> # Convert to SAT-like scale (200-800)
  >>> theta_to_scale(0.0, scale_min=200, scale_max=800)
  500.0

  >>> # Convert multiple values at once
  >>> import numpy as np
  >>> thetas = np.array([-4, -2, 0, 2, 4])
  >>> theta_to_scale(thetas)  # doctest: +SKIP
  array([  0.,  25.,  50.,  75., 100.])

  Notes
  -----
  - Values outside [theta_min, theta_max] are extrapolated linearly
  - The transformation preserves relative distances between ability levels
  - Common score scales: 0-100, 200-800 (SAT), 0-500 (TOEFL), etc.
  """
  if theta_max <= theta_min:
    msg = f"theta_max ({theta_max}) must be greater than theta_min ({theta_min})"
    raise ValueError(msg)
  if scale_max <= scale_min:
    msg = f"scale_max ({scale_max}) must be greater than scale_min ({scale_min})"
    raise ValueError(msg)

  # Linear transformation: score = a * theta + b
  # Solve for a and b using the two boundary conditions
  a = (scale_max - scale_min) / (theta_max - theta_min)
  b = scale_min - a * theta_min

  return a * theta + b


def scale_to_theta(
  score: float | npt.NDArray[numpy.floating[Any]],
  scale_min: float = 0,
  scale_max: float = 100,
  theta_min: float = -4,
  theta_max: float = 4,
) -> float | npt.NDArray[numpy.floating[Any]]:
  r"""Convert scores from a custom scale to theta values using linear transformation.

  This function transforms scores from any desired scale (e.g., 0-100, 200-800)
  back to the standard IRT theta scale (typically centered at 0).

  The inverse linear transformation is:

  .. math:: \theta = \frac{\text{score} - b}{a}

  where :math:`a` and :math:`b` are determined by the scale mapping.

  Parameters
  ----------
  score : float or numpy.ndarray
      Score value(s) on the custom scale.
  scale_min : float, optional
      Minimum value of the score scale. Default is 0.
  scale_max : float, optional
      Maximum value of the score scale. Default is 100.
  theta_min : float, optional
      Minimum theta value in the mapping. Default is -4.
  theta_max : float, optional
      Maximum theta value in the mapping. Default is 4.

  Returns
  -------
  float or numpy.ndarray
      The transformed theta value(s). Returns the same type as input.

  Examples
  --------
  >>> # Convert 0-100 score to theta
  >>> scale_to_theta(50.0)  # Average score
  0.0
  >>> scale_to_theta(75.0)  # High score
  2.0
  >>> scale_to_theta(25.0)  # Low score
  -2.0

  >>> # Convert from SAT-like scale (200-800)
  >>> scale_to_theta(500.0, scale_min=200, scale_max=800)
  0.0

  >>> # Convert multiple values at once
  >>> import numpy as np
  >>> scores = np.array([0, 25, 50, 75, 100])
  >>> scale_to_theta(scores)  # doctest: +SKIP
  array([-4., -2.,  0.,  2.,  4.])

  Notes
  -----
  - This is the inverse of :py:func:`theta_to_scale`
  - Useful for converting cutoff scores to theta values for classification
  - Preserves the relative ordering and distances of scores
  """
  if theta_max <= theta_min:
    msg = f"theta_max ({theta_max}) must be greater than theta_min ({theta_min})"
    raise ValueError(msg)
  if scale_max <= scale_min:
    msg = f"scale_max ({scale_max}) must be greater than scale_min ({scale_min})"
    raise ValueError(msg)

  # Inverse transformation: theta = (score - b) / a
  a = (scale_max - scale_min) / (theta_max - theta_min)
  b = scale_min - a * theta_min

  return (score - b) / a


def reliability(theta: float, items: npt.NDArray[numpy.floating[Any]]) -> float:
  r"""Compute test reliability [Thissen00]_.

  Test reliability is a measure of internal consistency for the test, similar to
  Cronbach's :math:`\alpha` in Classical Test Theory. Its value is always lower than 1,
  with values close to 1 indicating good reliability. If :math:`I(\theta) < 1`,
  :math:`Rel < 0` and in these cases it does not make sense, but usually the
  application of additional items solves this problem.

  .. math:: Rel = 1 - \frac{1}{I(\theta)}

  Parameters
  ----------
  theta : float
      An ability value.
  items : numpy.ndarray
      A matrix containing item parameters.

  Returns
  -------
  float
      The test reliability at `theta` for a test represented by `items`. Values
      range from negative infinity to just below 1, with values closer to 1
      indicating higher reliability.
  """
  return 1 - var(theta, items)


def max_info(a: float = 1, b: float = 0, c: float = 0, d: float = 1) -> float:
  r"""Return the :math:`\theta` value at which an item provides maximum information.

  For the 1-parameter and 2-parameter logistic models, this :math:`\theta` equals :math:`b`.
  In the 3-parameter and 4-parameter logistic models, however, this value is given by
  ([Magis13]_)

  .. math:: argmax_{\theta}I(\theta) = b + \frac{1}{a} log \left(\frac{x^* - c}{d - x^*}\right)

  where

  .. math::

    x^* = 2 \sqrt{\frac{-u}{3}} cos\left\{\frac{1}{3}acos\left(-\frac{v}{2}\sqrt{\frac{27}{-u^3}}\right)+
    \frac{4 \pi}{3}\right\} + 0.5

  .. math:: u = -\frac{3}{4} + \frac{c + d - 2cd}{2}

  .. math:: v = \frac{c + d - 1}{4}

  A few results can be seen in the plots below:

  .. plot::
      :caption: Item information curves for two distinct items. The point of maximum information denoted by a dot.

      import matplotlib.pyplot as plt
      from catsim.item_bank import ItemBank
      from catsim.plot import item_curve, PlotType
      n_items = 2
      item_bank = ItemBank.generate_item_bank(n_items)
      fig, axes = plt.subplots(n_items, 1)
      for idx, item in enumerate(item_bank.items):
        item_curve(item[0], item[1], item[2], item[3], ptype=PlotType.IIC, max_info=True, ax=axes[idx])
      plt.show()

  Parameters
  ----------
  a : float, optional
      Item discrimination parameter. Default is 1.
  b : float, optional
      Item difficulty parameter. Default is 0.
  c : float, optional
      Item pseudo-guessing parameter. Default is 0.
  d : float, optional
      Item upper asymptote. Default is 1.

  Returns
  -------
  float
      The :math:`\theta` value at which the item with the given parameters
      provides maximum information.
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


def max_info_hpc(items: npt.NDArray[numpy.floating[Any]]) -> npt.NDArray[numpy.floating[Any]]:
  """Parallelized version of :py:func:`max_info` using :py:mod:`numpy` and :py:mod:`numexpr`.

  This high-performance computing version efficiently computes the theta values of
  maximum information for multiple items simultaneously using vectorized operations.

  Parameters
  ----------
  items : numpy.ndarray
      Array containing item parameters with at least four columns [a, b, c, d, ...].

  Returns
  -------
  numpy.ndarray
      An array of theta values, one for each item, indicating where each item's
      information function reaches its maximum.
  """
  a, b, c, d = _split_params(items)

  if all(d == 1):
    if all(c == 0):
      return b
    return numexpr.evaluate("b + (1 / a) * log((1 + sqrt(1 + 8 * c)) / 2)", local_dict={"a": a, "b": b, "c": c})

  u = numexpr.evaluate("-(3 / 4) + ((c + d - 2 * c * d) / 2)", local_dict={"c": c, "d": d})
  v = numexpr.evaluate("(c + d - 1) / 4", local_dict={"c": c, "d": d})
  x_star = numexpr.evaluate(
    "2 * sqrt(-u / 3) * cos((1 / 3) * arccos(-(v / 2) * sqrt(27 / -(u ** 3))) + (4 * pi / 3)) + 0.5",
    local_dict={"u": u, "v": v, "pi": math.pi},
  )

  return numexpr.evaluate(
    "b + (1 / a) * log((x_star - c) / (d - x_star))",
    local_dict={"a": a, "b": b, "c": c, "d": d, "x_star": x_star},
  )


def log_likelihood(
  est_theta: float,
  response_vector: list[bool],
  administered_items: npt.NDArray[numpy.floating[Any]],
) -> float:
  r"""Compute the log-likelihood of an ability, given a response vector and the parameters of the answered items.

  The likelihood function of a given :math:`\theta` value given the answers to :math:`I`
  items is given by [Ayala2009]_:

  .. math:: L(X_{Ij} | \theta_j, a_I, b_I, c_I, d_I) = \prod_{i=1} ^ I P_{ij}(\theta)^{X_{ij}} Q_{ij}(\theta)^{1-X_{ij}}

  Finding the maximum of :math:`L(X_{Ij})` includes using the product rule of derivations.
  Since :math:`L(X_{Ij})` has :math:`j` parts, it can be quite complicated to do so. Also,
  for computational reasons, the product of probabilities can quickly tend to 0, so it is
  common to use the log-likelihood in maximization/minimization problems, transforming the
  product of probabilities into a sum of log probabilities:

   .. math::
      \log L(X_{Ij} | \theta_j, a_I, b_I, c_I, d_I) = \sum_{i=1} ^ I \left\lbrace x_{ij} \log P_{ij}(\theta)+
          (1 - x_{ij}) \log Q_{ij}(\theta) \right\rbrace

  where :math:`Q_{ij}(\theta) = 1 - P_{ij}(\theta)`.

  Parameters
  ----------
  est_theta : float
      Estimated ability value.
  response_vector : list[bool]
      A boolean list containing the response vector, where True indicates a correct
      response and False indicates an incorrect response.
  administered_items : numpy.ndarray
      A numpy array containing the parameters of the answered items.

  Returns
  -------
  float
      Log-likelihood of the given ability value, given the responses to the
      administered items. Higher values indicate better fit.

  Raises
  ------
  ValueError
      If response vector and administered items have different lengths or if the
      response vector contains non-Boolean elements.
  """
  if len(response_vector) != administered_items.shape[0]:
    msg = "Response vector and administered items must have the same number of items"
    raise ValueError(msg)
  if len(set(response_vector) - {True, False}) > 0:
    msg = "Response vector must contain only Boolean elements"
    raise ValueError(msg)
  ps = icc_hpc(est_theta, administered_items)
  result = numexpr.evaluate(
    "sum(where(response_vector, log(ps), log(1 - ps)))",
    local_dict={"response_vector": response_vector, "ps": ps},
  )
  return float(result)


def negative_log_likelihood(est_theta: float, *args: Any) -> float:
  """Compute the negative log-likelihood of an ability value for a response vector and parameters of administered items.

  This function is used by :py:mod:`scipy.optimize` optimization functions that tend to
  minimize values, instead of maximizing them. Since we want to maximize the log-likelihood,
  we minimize the negative log-likelihood. The value of :py:func:`negative_log_likelihood`
  is simply :math:`-` :py:func:`log_likelihood`.

  Parameters
  ----------
  est_theta : float
      Estimated ability value.
  *args : tuple
      Variable length argument list. Expected to contain:

      - args[0]: response_vector (list[bool]) - A Boolean list containing the response vector
      - args[1]: administered_items (numpy.ndarray) - A numpy array containing the parameters
        of the answered items

  Returns
  -------
  float
      Negative log-likelihood of a given ability value, given the responses to the
      administered items. Lower values indicate better fit.
  """
  return -log_likelihood(est_theta, args[0], args[1])


def normalize_item_bank(items: npt.NDArray[numpy.floating[Any]]) -> npt.NDArray[numpy.floating[Any]]:
  r"""Normalize an item matrix so that it conforms to the standard used by catsim.

  The item matrix must have dimension :math:`n \times 4`, in which column 0 represents item
  discrimination, column 1 represents item difficulty, column 2 represents the pseudo-guessing
  parameter and column 3 represents the item upper asymptote.

  This function automatically expands item matrices with fewer than 4 columns:

  - 1 column (1PL): Assumed to be difficulty. Discrimination=1, c=0, d=1 are added.
  - 2 columns (2PL): Assumed to be [discrimination, difficulty]. c=0, d=1 are added.
  - 3 columns (3PL): Assumed to be [discrimination, difficulty, pseudo-guessing]. d=1 is added.
  - 4 columns (4PL): Already complete, returned as-is.

  Parameters
  ----------
  items : numpy.ndarray
      The item matrix with 1, 2, 3, or 4 columns.

  Returns
  -------
  numpy.ndarray
      An :math:`n \times 4` item matrix conforming to the 4-parameter logistic model,
      with columns [a, b, c, d].
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


def validate_item_bank(items: npt.NDArray[numpy.floating[Any]], raise_err: bool = False) -> None:
  r"""Validate the shape and parameters in the item matrix to ensure it conforms to catsim standards.

  The item matrix must have dimension :math:`n \times 4`, in which column 0 represents item
  discrimination, column 1 represents item difficulty, column 2 represents the pseudo-guessing
  parameter and column 3 represents the item upper asymptote.

  The item matrix must have at least one line (representing an item) and exactly four columns,
  representing the discrimination, difficulty, pseudo-guessing and upper asymptote parameters
  (:math:`a`, :math:`b`, :math:`c` and :math:`d`), respectively. The item matrix is considered
  valid if, for all items in the matrix, :math:`a > 0 \wedge 0 \leq c \leq 1 \wedge 0 \leq d \leq 1`.

  Parameters
  ----------
  items : numpy.ndarray
      The item matrix to validate.
  raise_err : bool, optional
      Whether to raise an error in case the validation fails (True) or just print
      the error message to standard output (False). Default is False.

  Raises
  ------
  TypeError
      If items is not a numpy.ndarray.
  ValueError
      If raise_err is True and the item matrix does not meet validation criteria.
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
