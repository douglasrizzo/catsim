"""Item response theory quantities of the four-parameter logistic (4PL) model.

The functions take one item as a row ``[a, b, c, d]`` of shape ``(4,)``, or several as a matrix of
such rows, shape ``(n, 4)``: discrimination, difficulty, pseudo-guessing and upper asymptote.
:py:class:`catsim.item_bank.ItemBank` builds and validates such a matrix.
"""

import math
from collections.abc import Sequence
from typing import overload

import numpy
from scipy import stats
from scipy.special import expit, log_expit

from .typing import FloatArray

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


def _split_params(items: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
  """Return the a, b, c and d parameters of a row or matrix of items, each of shape ``items.shape[:-1]``."""
  return items[..., 0], items[..., 1], items[..., 2], items[..., 3]


def _logit(theta: float | FloatArray, items: FloatArray) -> FloatArray:
  """Return a (theta - b) for every ability against every item, of shape ``numpy.shape(theta) + items.shape[:-1]``."""
  a, b, _, _ = _split_params(items)
  thetas = numpy.asarray(theta, dtype=float)
  return a * (thetas.reshape(thetas.shape + (1,) * (items.ndim - 1)) - b)


def icc(theta: float | FloatArray, items: FloatArray) -> FloatArray:
  r"""Return the probability of a correct response to each item, its item characteristic curve.

  Under the four-parameter logistic model :cite:p:`Magis13`, an examinee of ability :math:`\theta`
  answers item :math:`i` correctly with probability

  .. math:: P_i(\theta) = c_i + \frac{d_i - c_i}{1 + e^{-a_i(\theta - b_i)}}

  The probability rises with :math:`\theta` from the pseudo-guessing parameter :math:`c_i` to the
  upper asymptote :math:`d_i`. It rises fastest at the difficulty :math:`b_i`, where it equals
  :math:`(c_i + d_i) / 2`, and the discrimination :math:`a_i` sets how fast.

  Parameters
  ----------
  theta : float or numpy.ndarray
      An ability, or an array of abilities.
  items : numpy.ndarray
      One item ``[a, b, c, d]``, shape ``(4,)``, or a matrix of items, shape ``(n, 4)``.

  Returns
  -------
  numpy.ndarray
      Every ability against every item, of shape ``numpy.shape(theta) + items.shape[:-1]``: a float
      ``theta`` gives ``(n,)`` for ``n`` items, ``k`` abilities give ``(k, n)``, and a float
      ``theta`` with one row gives a 0-d result.

  Examples
  --------
  >>> import numpy
  >>> items = numpy.array([[1.0, 0.0, 0.0, 1.0], [1.5, 1.0, 0.2, 1.0]])
  >>> icc(0.0, items).round(4)
  array([0.5   , 0.3459])
  >>> icc(numpy.array([-1.0, 0.0, 1.0]), items).shape
  (3, 2)
  """
  _, _, c, d = _split_params(items)
  return c + (d - c) * expit(_logit(theta, items))


def information(theta: float | FloatArray, items: FloatArray) -> FloatArray:
  r"""Return the Fisher information each item gives about the ability.

  The information of item :math:`i` at :math:`\theta` :cite:p:`Ayala2009,Magis13` is

  .. math:: I_i(\theta) = \frac{a_i^2 (P_i - c_i)^2 (d_i - P_i)^2}{(d_i - c_i)^2 P_i (1 - P_i)}

  where :math:`P_i = P_i(\theta)` is the :py:func:`icc`. An item is most informative near its difficulty, at
  :py:func:`max_info_theta`; a 2PL item (:math:`c_i = 0`, :math:`d_i = 1`) gives :math:`a_i^2 / 4`
  at :math:`\theta = b_i`. Far from the difficulty the information tends to 0 and stays finite.

  Parameters
  ----------
  theta : float or numpy.ndarray
      An ability, or an array of abilities.
  items : numpy.ndarray
      One item ``[a, b, c, d]``, shape ``(4,)``, or a matrix of items, shape ``(n, 4)``.

  Returns
  -------
  numpy.ndarray
      Every ability against every item, of shape ``numpy.shape(theta) + items.shape[:-1]``: a float
      ``theta`` gives ``(n,)`` for ``n`` items, ``k`` abilities give ``(k, n)``, and a float
      ``theta`` with one row gives a 0-d result.

  Examples
  --------
  >>> import numpy
  >>> information(0.0, numpy.array([[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]]))
  array([0.25, 1.  ])
  """
  a, _, c, d = _split_params(items)
  z = _logit(theta, items)
  # With L the logistic term, P - c = (d - c) L and 1 - P = (1 - d) + (d - c)(1 - L), so
  # I = a^2 L (1 - L) [(d - c) L / P] [(d - c)(1 - L) / (1 - P)]. This form never computes 1 - P from a
  # P that has rounded to 1, which would divide 0 by 0 far from b.
  logistic, complement = expit(z), expit(-z)
  product = logistic * complement
  above_floor = (d - c) * logistic
  below_ceiling = (d - c) * complement
  # where L (1 - L) underflows to 0, so does the information, and the ratios would be 0 / 0
  with numpy.errstate(divide="ignore", invalid="ignore"):
    values = a**2 * product * (above_floor / (c + above_floor)) * (below_ceiling / ((1 - d) + below_ceiling))
  return numpy.where(product > 0, values, 0.0)


@overload
def total_information(theta: float, items: FloatArray) -> float: ...
@overload
def total_information(theta: FloatArray, items: FloatArray) -> FloatArray: ...
def total_information(theta: float | FloatArray, items: FloatArray) -> float | FloatArray:
  r"""Return the test information of a set of items, the sum of their information.

  .. math:: I(\theta) = \sum_{i=1}^{n} I_i(\theta)

  where :math:`I_i` is the item :py:func:`information` :cite:p:`Ayala2009`. The more information
  at :math:`\theta`, the more precise an ability estimate near it: its inverse is the variance of
  the estimate (:py:func:`var`). Items without information, or no items at all, give 0.

  Parameters
  ----------
  theta : float or numpy.ndarray
      An ability, or an array of abilities.
  items : numpy.ndarray
      The items, a matrix of shape ``(n, 4)``.

  Returns
  -------
  float or numpy.ndarray
      One value per ability: a float for a float ``theta``, otherwise an array of the shape of
      ``theta``.

  Examples
  --------
  >>> import numpy
  >>> total_information(0.0, numpy.array([[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]]))
  1.25
  """
  values = numpy.sum(information(theta, items), axis=-1)
  return values if numpy.ndim(theta) else float(values)


def var(theta: float, items: FloatArray) -> float:
  r"""Return the variance of the ability estimate at ``theta``, the inverse of the test information.

  .. math:: Var(\theta) = \frac{1}{I(\theta)}

  where :math:`I` is the :py:func:`total_information`. This is the asymptotic variance of the
  maximum likelihood estimate :cite:p:`Ayala2009`; the lower it is, the more precise the estimate.

  Parameters
  ----------
  theta : float
      An ability.
  items : numpy.ndarray
      The administered items, a matrix of shape ``(n, 4)``.

  Returns
  -------
  float
      The variance, infinite when the items give no information at ``theta``.

  Examples
  --------
  >>> import numpy
  >>> var(0.0, numpy.array([[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]]))
  0.8
  """
  test_information = total_information(theta, items)
  return math.inf if test_information == 0 else 1 / test_information


def see(theta: float, items: FloatArray) -> float:
  r"""Return the standard error of estimation at ``theta``, the square root of the variance.

  .. math:: SEE(\theta) = \sqrt{\frac{1}{I(\theta)}}

  where :math:`I` is the :py:func:`total_information` :cite:p:`Ayala2009`. It is in the units of
  the ability scale, which makes it easier to read than the :py:func:`var`.

  Parameters
  ----------
  theta : float
      An ability.
  items : numpy.ndarray
      The administered items, a matrix of shape ``(n, 4)``.

  Returns
  -------
  float
      The standard error, infinite when the items give no information at ``theta``.

  Examples
  --------
  >>> import numpy
  >>> see(0.0, numpy.array([[1.0, 0.0, 0.0, 1.0]]))
  2.0
  """
  return math.sqrt(var(theta, items))


def confidence_interval(theta: float, items: FloatArray, confidence: float = 0.95) -> tuple[float, float]:
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
  items : numpy.ndarray
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
  theta: float | FloatArray,
  scale_min: float = 0,
  scale_max: float = 100,
  theta_min: float = -4,
  theta_max: float = 4,
) -> float | FloatArray:
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

  Raises
  ------
  ValueError
      If ``theta_max`` is not greater than ``theta_min``, or ``scale_max`` is not greater than
      ``scale_min``.

  Notes
  -----
  - Values outside [theta_min, theta_max] are extrapolated linearly
  - The transformation preserves relative distances between ability levels
  - Common score scales: 0-100, 200-800 (SAT), 0-500 (TOEFL), etc.

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
  score: float | FloatArray,
  scale_min: float = 0,
  scale_max: float = 100,
  theta_min: float = -4,
  theta_max: float = 4,
) -> float | FloatArray:
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

  Raises
  ------
  ValueError
      If ``theta_max`` is not greater than ``theta_min``, or ``scale_max`` is not greater than
      ``scale_min``.

  Notes
  -----
  - This is the inverse of :py:func:`theta_to_scale`
  - Useful for converting cutoff scores to theta values for classification
  - Preserves the relative ordering and distances of scores

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


def reliability(theta: float, items: FloatArray) -> float:
  r"""Return the test reliability at ``theta``.

  .. math:: Rel(\theta) = 1 - \frac{1}{I(\theta)}

  where :math:`I` is the :py:func:`total_information` :cite:p:`Thissen00`. Like Cronbach's
  :math:`\alpha` in classical test theory, values close to 1 mean a precise test. It is below 1
  always and negative when :math:`I(\theta) < 1`, which more items usually remedy.

  Parameters
  ----------
  theta : float
      An ability.
  items : numpy.ndarray
      The administered items, a matrix of shape ``(n, 4)``.

  Returns
  -------
  float
      The reliability, negative infinity when the items give no information at ``theta``.

  Examples
  --------
  >>> import numpy
  >>> reliability(0.0, numpy.array([[2.0, 0.0, 0.0, 1.0]] * 4))
  0.75
  """
  return 1 - var(theta, items)


def max_info_theta(items: FloatArray) -> FloatArray:
  r"""Return the ability at which each item gives the most information.

  For an item with :math:`d_i = 1` the peak is at :cite:p:`Magis13`

  .. math:: \theta^*_i = b_i + \frac{1}{a_i} \log \frac{1 + \sqrt{1 + 8 c_i}}{2}

  which is the difficulty :math:`b_i` for a 1PL or 2PL item. Otherwise it is

  .. math:: \theta^*_i = b_i + \frac{1}{a_i} \log \frac{x^*_i - c_i}{d_i - x^*_i}

  where

  .. math::

    x^*_i = 2 \sqrt{\frac{-u_i}{3}} \cos\left\{\frac{1}{3} \arccos\left(-\frac{v_i}{2}
    \sqrt{\frac{27}{-u_i^3}}\right) + \frac{4 \pi}{3}\right\} + \frac{1}{2}, \quad
    u_i = -\frac{3}{4} + \frac{c_i + d_i - 2 c_i d_i}{2}, \quad v_i = \frac{c_i + d_i - 1}{4}

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
      plt.tight_layout()
      plt.show()

  Parameters
  ----------
  items : numpy.ndarray
      One item ``[a, b, c, d]``, shape ``(4,)``, or a matrix of items, shape ``(n, 4)``.

  Returns
  -------
  numpy.ndarray
      A new array of shape ``items.shape[:-1]``, one ability per item.

  Examples
  --------
  >>> import numpy
  >>> max_info_theta(numpy.array([[1.0, 0.5, 0.0, 1.0], [1.0, 0.0, 0.2, 1.0]])).round(4)
  array([0.5   , 0.2671])
  """
  a, b, c, d = _split_params(items)
  three_pl = b + numpy.log((1 + numpy.sqrt(1 + 8 * c)) / 2) / a
  u = -(3 / 4) + (c + d - 2 * c * d) / 2
  v = (c + d - 1) / 4
  x_star = 2 * numpy.sqrt(-u / 3) * numpy.cos(numpy.arccos(-(v / 2) * numpy.sqrt(27 / -(u**3))) / 3 + 4 * math.pi / 3)
  x_star += 0.5
  four_pl = b + numpy.log((x_star - c) / (d - x_star)) / a
  return numpy.where(d == 1, three_pl, four_pl)


def info_integral(lower: float, upper: float, items: FloatArray) -> FloatArray:
  r"""Return the integral of each item's information between two abilities.

  Since :math:`dP/d\theta = a (P - c)(d - P) / (d - c)`, the item :py:func:`information`
  :math:`I(\theta) = (dP/d\theta)^2 / [P (1 - P)]` integrates in closed form, with :math:`P` the
  :py:func:`icc`:

  .. math::

      \int_{\theta_L}^{\theta_U} I(\theta)\, d\theta =
      \frac{a}{d - c} \Big[ P - c\, d \log P + (1 - c)(1 - d) \log(1 - P) \Big]_{P(\theta_L)}^{P(\theta_U)}

  The bounds may be infinite. Over the whole ability scale, a 2PL item's integral is :math:`a`.

  Parameters
  ----------
  lower : float
      Lower integration bound, possibly ``-inf``.
  upper : float
      Upper integration bound, possibly ``inf``.
  items : numpy.ndarray
      One item ``[a, b, c, d]``, shape ``(4,)``, or a matrix of items, shape ``(n, 4)``.

  Returns
  -------
  numpy.ndarray
      The integrals over ``[lower, upper]``, of shape ``items.shape[:-1]``.

  Examples
  --------
  >>> import numpy
  >>> info_integral(-numpy.inf, numpy.inf, numpy.array([[0.7, -1.0, 0.0, 1.0], [1.3, 0.0, 0.0, 1.0]]))
  array([0.7, 1.3])
  """
  a, _, c, d = _split_params(items)
  log_p_weight, log_q_weight = c * d, (1 - c) * (1 - d)

  def antiderivative(theta: float) -> FloatArray:
    p = icc(theta, items)
    # a zero weight drops its logarithm, which is -inf at an infinite bound
    with numpy.errstate(divide="ignore", invalid="ignore"):
      log_p_term = numpy.where(log_p_weight > 0, log_p_weight * numpy.log(p), 0.0)
      log_q_term = numpy.where(log_q_weight > 0, log_q_weight * numpy.log1p(-p), 0.0)
    return a / (d - c) * (p - log_p_term + log_q_term)

  return antiderivative(upper) - antiderivative(lower)


@overload
def log_likelihood(theta: float, responses: Sequence[bool], items: FloatArray) -> float: ...
@overload
def log_likelihood(theta: FloatArray, responses: Sequence[bool], items: FloatArray) -> FloatArray: ...
def log_likelihood(theta: float | FloatArray, responses: Sequence[bool], items: FloatArray) -> float | FloatArray:
  r"""Return the log-likelihood of each ability, given the responses to a set of items.

  .. math::

      \log L(\theta) = \sum_{i=1}^{n} \left[ u_i \log P_i(\theta) + (1 - u_i) \log Q_i(\theta) \right]

  where :math:`u_i` is 1 for a correct response and 0 otherwise, :math:`P_i` is the :py:func:`icc`
  and :math:`Q_i = 1 - P_i` :cite:p:`Ayala2009`. Both logarithms are exact at any distance from the
  difficulty: far from it, a correct guess costs :math:`\log c_i` and a slip
  :math:`\log(1 - d_i)`. With no items the log-likelihood is 0.

  Parameters
  ----------
  theta : float or numpy.ndarray
      An ability, or an array of abilities.
  responses : Sequence[bool]
      One response per item, True for a correct one.
  items : numpy.ndarray
      The answered items, a matrix of shape ``(n, 4)`` whose items satisfy
      :math:`0 \leq c < d \leq 1`, as :py:class:`~catsim.item_bank.ItemBank` validates.

  Returns
  -------
  float or numpy.ndarray
      One value per ability: a float for a float ``theta``, otherwise an array of the shape of
      ``theta``.

  Raises
  ------
  ValueError
      If ``responses`` does not hold one Boolean per item.

  Examples
  --------
  >>> import numpy
  >>> round(log_likelihood(0.0, [True, False], numpy.array([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])), 4)
  -1.3863
  """
  validate_response_vector(responses, items.shape[0])
  _, _, c, d = _split_params(items)
  z = _logit(theta, items)
  # c = 0 and d = 1 give log(0) = -inf, which logaddexp handles exactly
  with numpy.errstate(divide="ignore"):
    log_c, log_1md, log_dmc = numpy.log(c), numpy.log1p(-d), numpy.log(d - c)
  log_p = numpy.logaddexp(log_c, log_dmc + log_expit(z))
  log_q = numpy.logaddexp(log_1md, log_dmc + log_expit(-z))
  values = numpy.sum(numpy.where(numpy.asarray(responses, dtype=bool), log_p, log_q), axis=-1)
  return values if numpy.ndim(theta) else float(values)


def validate_response_vector(responses: Sequence[bool], n_items: int) -> None:
  """Check that ``responses`` holds one Boolean answer per administered item.

  Parameters
  ----------
  responses : Sequence[bool]
      The examinee's answers, where True indicates a correct response.
  n_items : int
      Number of administered items the answers belong to.

  Raises
  ------
  ValueError
      If ``responses`` does not have ``n_items`` elements, or if it contains non-Boolean elements.
  """
  if len(responses) != n_items:
    msg = f"responses and items must have the same length, got {len(responses)} and {n_items}"
    raise ValueError(msg)
  if len(set(responses) - {True, False}) > 0:
    msg = "responses must contain only Boolean elements"
    raise ValueError(msg)
