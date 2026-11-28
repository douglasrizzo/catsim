r"""The performance measures of a simulation study, each with its Monte Carlo standard error.

A simulation estimates every measure from a finite number of sessions, so a rerun with another seed
gives another value. Each function here returns a :class:`Metric`: the value, and its Monte Carlo
standard error, an estimate of the standard deviation of that value across such reruns
:cite:p:`Morris2019`. A difference between two designs that is small next to their errors cannot be
told from noise. :class:`~catsim.SimulationResult` computes these measures for a run, overall and by
decile of the true ability.

The errors treat the sessions as independent draws, as they are when no session sees exposure rates:
every examinee draws from streams of their own (ADR-0013). Under exposure feedback each session
depends on the ones before it, so the errors are approximations, and replications of the run give
the exact spread (ADR-0016).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import numpy
from irtlib.typing import FloatArray

if TYPE_CHECKING:
  import numpy.typing as npt
  from irtlib.typing import IntArray


class Metric[T: (float, FloatArray)](NamedTuple):
  """A performance measure of a simulation run, with its Monte Carlo standard error.

  A metric unpacks as a pair, ``value, mc_se = metrics.bias(true_thetas, thetas)``. A measure with
  one value per item, such as the exposure rates, holds one array of values and one of errors.

  Parameters
  ----------
  value : float or numpy.ndarray
      The measure's value over the run's sessions.
  mc_se : float or numpy.ndarray
      Its Monte Carlo standard error, NaN where the sessions give no estimate of it, as a single
      session cannot.

  Examples
  --------
  >>> from catsim.metrics import bias
  >>> value, mc_se = bias([0.0, 0.0, 0.0], [0.5, -0.5, 1.5])
  >>> value, round(mc_se, 4)
  (0.5, 0.5774)
  """

  value: T
  mc_se: T


def _vector(name: str, values: npt.ArrayLike) -> FloatArray:
  """Return one value per session as a float array, checking that there is at least one and no NaN.

  Parameters
  ----------
  name : str
      The parameter's name, for the error messages.
  values : array-like
      The values.

  Returns
  -------
  numpy.ndarray
      The values as a one-dimensional float array.

  Raises
  ------
  TypeError
      If ``values`` is not one-dimensional or holds values that are not real numbers.
  ValueError
      If ``values`` is empty or holds NaN.
  """
  array = numpy.asarray(values)
  if array.ndim != 1:
    msg = f"{name} must be one-dimensional, one value per session, got shape {array.shape}"
    raise TypeError(msg)
  # Booleans would be read as the numbers 0 and 1.
  if array.dtype.kind not in "iuf":
    msg = f"{name} holds {array.dtype} values; they must be real numbers"
    raise TypeError(msg)
  if array.size == 0:
    msg = f"{name} cannot be empty"
    raise ValueError(msg)
  array = array.astype(float)
  if numpy.isnan(array).any():
    position = int(numpy.flatnonzero(numpy.isnan(array))[0])
    msg = f"{name}[{position}] is nan; a metric takes one number per session, with no missing ones"
    raise ValueError(msg)
  return array


def _errors(true_thetas: npt.ArrayLike, thetas: npt.ArrayLike) -> FloatArray:
  """Return each session's estimate minus its true ability.

  Parameters
  ----------
  true_thetas : array-like
      The true abilities, finite.
  thetas : array-like
      The estimates, one per true ability, finite.

  Returns
  -------
  numpy.ndarray
      The errors.

  Raises
  ------
  ValueError
      If the two differ in length, which would otherwise broadcast a single value against the other,
      or hold a value that is not finite.
  """
  actual = _vector("true_thetas", true_thetas)
  estimated = _vector("thetas", thetas)
  if len(actual) != len(estimated):
    msg = (
      f"true_thetas has {len(actual)} entries and thetas {len(estimated)}; there must be one estimate per true ability"
    )
    raise ValueError(msg)
  for name, array in (("true_thetas", actual), ("thetas", estimated)):
    if not numpy.isfinite(array).all():
      position = int(numpy.flatnonzero(~numpy.isfinite(array))[0])
      msg = f"{name}[{position}] is {array[position].item()!r}; abilities and their estimates must be finite"
      raise ValueError(msg)
  return estimated - actual


def mean(values: npt.ArrayLike) -> Metric[float]:
  r"""Return the mean of a statistic over a run's sessions, with its Monte Carlo standard error.

  .. math::

     \bar{x} = \frac{1}{N} \sum_{j=1}^{N} x_j, \qquad
     \widehat{\mathrm{SE}}(\bar{x}) = \sqrt{\frac{1}{N(N-1)} \sum_{j=1}^{N} (x_j - \bar{x})^2}

  where :math:`x_j` is session :math:`j`'s value and :math:`N` the number of sessions. This is the
  Monte Carlo standard error of a mean, as Morris, White & Crowther give it for the bias
  (:cite:t:`Morris2019`, Table 6) and Siepe et al. for the mean of any statistic
  (:cite:t:`Siepe2024`, Table 3). The mean test length and the mean standard error of a run are
  such means.

  Parameters
  ----------
  values : array-like
      One value per session, such as its test length or the standard error of its final estimate.

  Returns
  -------
  Metric
      The mean and its Monte Carlo standard error. The error is NaN for a single session, whose
      values have no spread to estimate it from, and for values that are not all finite, such as
      the infinite standard error of a session that ended before its first response, whose mean is
      infinite.

  Raises
  ------
  TypeError
      If ``values`` is not one-dimensional or holds values that are not real numbers.
  ValueError
      If ``values`` is empty or holds NaN.

  Examples
  --------
  Two tests of 10 and 14 items: their deviations from the mean, 12, are -2 and 2, so the error is
  :math:`\sqrt{8 / (2 \cdot 1)} = 2`.

  >>> from catsim.metrics import mean
  >>> mean([10, 14])
  Metric(value=12.0, mc_se=2.0)
  """
  array = _vector("values", values)
  value = float(numpy.mean(array))
  if array.size < 2 or not numpy.isfinite(array).all():  # noqa: PLR2004 - a spread needs two values
    return Metric(value, math.nan)
  return Metric(value, math.sqrt(float(numpy.var(array, ddof=1)) / array.size))


def bias(true_thetas: npt.ArrayLike, thetas: npt.ArrayLike) -> Metric[float]:
  r"""Return the mean signed error of the ability estimates :cite:p:`Chang2001`, with its Monte Carlo standard error.

  .. math::

     \mathrm{Bias} = \frac{1}{N} \sum_{j=1}^{N} e_j, \qquad
     \widehat{\mathrm{SE}}(\mathrm{Bias}) = \sqrt{\frac{1}{N(N-1)} \sum_{j=1}^{N} (e_j - \mathrm{Bias})^2}

  where :math:`e_j = \hat{\theta}_j - \theta_j` is examinee :math:`j`'s estimate minus their true
  ability and :math:`N` the number of examinees. A positive bias means the estimates overestimate
  ability on average, a negative one that they underestimate it. The error is that of the bias in
  :cite:t:`Morris2019` (Table 6), with :math:`e_j` in place of :math:`\hat{\theta}_i - \theta`,
  since each examinee has their own true ability: the error of the mean of the errors
  (:func:`mean`).

  Parameters
  ----------
  true_thetas : array-like
      The true abilities, finite.
  thetas : array-like
      The estimated abilities, one per true ability, finite.

  Returns
  -------
  Metric
      The bias and its Monte Carlo standard error, NaN for a single examinee.

  Raises
  ------
  TypeError
      If either input is not one-dimensional or holds values that are not real numbers.
  ValueError
      If the inputs are empty, differ in length or hold a value that is not finite.

  Examples
  --------
  >>> from catsim.metrics import bias
  >>> bias([1.0, -1.0], [0.0, 6.0])
  Metric(value=3.0, mc_se=4.0)
  """
  return mean(_errors(true_thetas, thetas))


def mse(true_thetas: npt.ArrayLike, thetas: npt.ArrayLike) -> Metric[float]:
  r"""Return the mean squared error of the ability estimates :cite:p:`Chang2001`, with its Monte Carlo standard error.

  .. math::

     \mathrm{MSE} = \frac{1}{N} \sum_{j=1}^{N} e_j^2, \qquad
     \widehat{\mathrm{SE}}(\mathrm{MSE}) = \sqrt{\frac{1}{N(N-1)} \sum_{j=1}^{N} (e_j^2 - \mathrm{MSE})^2}

  where :math:`e_j = \hat{\theta}_j - \theta_j` is examinee :math:`j`'s estimate minus their true
  ability and :math:`N` the number of examinees. Lower values mean more accurate estimates; the MSE
  is the squared :func:`bias` plus the variance of the errors. The error is that of the MSE in
  :cite:t:`Morris2019` (Table 6), with :math:`e_j` in place of :math:`\hat{\theta}_i - \theta`: the
  error of the mean of the squared errors (:func:`mean`).

  Parameters
  ----------
  true_thetas : array-like
      The true abilities, finite.
  thetas : array-like
      The estimated abilities, one per true ability, finite.

  Returns
  -------
  Metric
      The mean squared error and its Monte Carlo standard error, NaN for a single examinee.

  Raises
  ------
  TypeError
      If either input is not one-dimensional or holds values that are not real numbers.
  ValueError
      If the inputs are empty, differ in length or hold a value that is not finite.

  Examples
  --------
  >>> from catsim.metrics import mse
  >>> mse([1.0, -1.0], [0.0, 6.0])
  Metric(value=25.0, mc_se=24.0)
  """
  errors = _errors(true_thetas, thetas)
  return mean(errors * errors)


def rmse(true_thetas: npt.ArrayLike, thetas: npt.ArrayLike) -> Metric[float]:
  r"""Return the root mean squared error of the ability estimates :cite:p:`Bar10`, with its Monte Carlo standard error.

  .. math::

     \mathrm{RMSE} = \sqrt{\mathrm{MSE}}, \qquad
     \widehat{\mathrm{SE}}(\mathrm{RMSE}) = \frac{\widehat{\mathrm{SE}}(\mathrm{MSE})}{2\,\mathrm{RMSE}}

  where :math:`\mathrm{MSE}` and its error are those of :func:`mse`. The RMSE is in the units of the
  ability scale. Its error follows from the MSE's by the delta method, since the derivative of
  :math:`\sqrt{m}` is :math:`1 / (2\sqrt{m})`; it is the error :cite:t:`Siepe2024` give for the
  RMSE (Table 3). It is 0 when every estimate equals its true ability.

  Parameters
  ----------
  true_thetas : array-like
      The true abilities, finite.
  thetas : array-like
      The estimated abilities, one per true ability, finite.

  Returns
  -------
  Metric
      The root mean squared error and its Monte Carlo standard error, NaN for a single examinee.

  Raises
  ------
  TypeError
      If either input is not one-dimensional or holds values that are not real numbers.
  ValueError
      If the inputs are empty, differ in length or hold a value that is not finite.

  Examples
  --------
  >>> from catsim.metrics import rmse
  >>> rmse([1.0, -1.0], [0.0, 6.0])
  Metric(value=5.0, mc_se=2.4)
  """
  squared = mse(true_thetas, thetas)
  value = math.sqrt(squared.value)
  # With every error 0, the MSE's error is 0 too (or NaN for one examinee), and so is the RMSE's.
  return Metric(value, squared.mc_se / (2.0 * value) if value > 0 else squared.mc_se)


def exposure_rates(counts: npt.ArrayLike, n_sessions: int) -> Metric[FloatArray]:
  r"""Return each item's exposure rate, the share of a run's sessions given it, with its Monte Carlo standard error.

  .. math::

     r_i = \frac{c_i}{N}, \qquad \widehat{\mathrm{SE}}(r_i) = \sqrt{\frac{r_i (1 - r_i)}{N}}

  where :math:`c_i` is the number of sessions that administered item :math:`i` and :math:`N` the
  number of sessions. A session gives an item at most once, so a rate is a proportion of the
  sessions, and its error is the binomial one that :cite:t:`Morris2019` (Table 6) give for coverage
  and rejection rates. The error is 0 for an item that no session, or every session, was given.
  The proposal counts of a run give its proposal rates the same way.

  Parameters
  ----------
  counts : array-like
      Each item's count over the run: the sessions that administered it, or that proposed it.
      Integers from 0 to ``n_sessions``.
  n_sessions : int
      The number of sessions in the run, positive.

  Returns
  -------
  Metric
      One rate per item, and one Monte Carlo standard error per item, as arrays.

  Raises
  ------
  TypeError
      If ``counts`` is not a one-dimensional array of integers, or ``n_sessions`` is not an integer.
  ValueError
      If ``counts`` is empty, ``n_sessions`` is not positive, or a count is negative or above
      ``n_sessions``.

  Examples
  --------
  >>> from catsim.metrics import exposure_rates
  >>> rates, errors = exposure_rates([4, 1, 0], n_sessions=4)
  >>> rates.tolist(), errors.round(4).tolist()
  ([1.0, 0.25, 0.0], [0.0, 0.2165, 0.0])
  """
  if isinstance(n_sessions, (bool, numpy.bool_)) or not isinstance(n_sessions, (int, numpy.integer)):
    msg = f"n_sessions is {n_sessions!r}; it must be an integer"
    raise TypeError(msg)
  if n_sessions < 1:
    msg = f"n_sessions is {n_sessions}; a run has at least one session"
    raise ValueError(msg)
  array = numpy.asarray(counts)
  # Before the type check: an empty list has NumPy's float dtype, and a bank has at least one item.
  if array.size == 0:
    msg = "counts is empty; there is one count per item of the bank"
    raise ValueError(msg)
  if array.ndim != 1 or array.dtype.kind not in "iu":
    msg = f"counts must be a one-dimensional array of integers, one per item, got {array.dtype} of shape {array.shape}"
    raise TypeError(msg)
  outside = (array < 0) | (array > n_sessions)
  if outside.any():
    item = int(numpy.flatnonzero(outside)[0])
    msg = f"counts[{item}] is {array[item].item()}; a count must be between 0 and the {n_sessions} sessions"
    raise ValueError(msg)
  rates = array / n_sessions
  return Metric(rates, numpy.sqrt(rates * (1.0 - rates) / n_sessions))


def _overlap(rates: FloatArray, test_size: int) -> float:
  """Return the large-sample test overlap rate of tests of ``test_size`` items, from the items' exposure rates.

  Parameters
  ----------
  rates : numpy.ndarray
      Each item's exposure rate.
  test_size : int
      The number of items in each test.

  Returns
  -------
  float
      :math:`(M / L) S_r^2 + L / M`, as :func:`overlap_rate` defines it.
  """
  bank_size = rates.shape[0]
  return (bank_size / test_size) * float(numpy.var(rates)) + (test_size / bank_size)


def _test_matrix(administered_items: npt.ArrayLike, n_items: int) -> IntArray:
  """Return the items of a run's fixed-length tests as an integer matrix, checking them.

  Parameters
  ----------
  administered_items : array-like
      One row per session, with the indices of the items it administered.
  n_items : int
      The number of items in the bank.

  Returns
  -------
  numpy.ndarray
      The tests, one row per session.

  Raises
  ------
  TypeError
      If the tests are not a two-dimensional array of integers, or ``n_items`` is not an integer.
  ValueError
      If there is no session or no item per session, an index is outside the bank, or a session
      repeats an item.
  """
  if isinstance(n_items, (bool, numpy.bool_)) or not isinstance(n_items, (int, numpy.integer)):
    msg = f"n_items is {n_items!r}; it must be an integer"
    raise TypeError(msg)
  tests = numpy.asarray(administered_items)
  # Before the shape check: an empty list has one dimension and NumPy's float dtype, and is still no sessions.
  if tests.size == 0:
    msg = f"administered_items has shape {tests.shape}; the overlap rate needs at least one session of one item or more"
    raise ValueError(msg)
  if tests.ndim != 2 or tests.dtype.kind not in "iu":  # noqa: PLR2004 - a matrix
    msg = (
      "administered_items must be a two-dimensional array of item indices, one row per session and one column per "
      f"position of the fixed-length tests, got {tests.dtype} of shape {tests.shape}"
    )
    raise TypeError(msg)
  outside = (tests < 0) | (tests >= n_items)
  if outside.any():
    row, column = (int(index) for index in numpy.argwhere(outside)[0])
    msg = f"administered_items[{row}, {column}] is {tests[row, column].item()}; an item index is 0 to {n_items - 1}"
    raise ValueError(msg)
  repeated = (numpy.diff(numpy.sort(tests, axis=1), axis=1) == 0).any(axis=1)
  if repeated.any():
    row = int(numpy.flatnonzero(repeated)[0])
    msg = f"administered_items[{row}] gives an item twice; a session administers each item at most once"
    raise ValueError(msg)
  return tests


def overlap_rate(administered_items: npt.ArrayLike, n_items: int) -> Metric[float]:
  r"""Return the test overlap rate of a run's fixed-length tests, with its Monte Carlo standard error.

  The test overlap rate is the share of items that the tests of two examinees have in common,
  averaged over every pair of examinees :cite:p:`Chen2003`. For :math:`N` tests of :math:`L` items
  each from a bank of :math:`M` items, with exposure rates :math:`r_i`, its large-sample value is

  .. math:: T = \frac{1}{L} \sum_{i=1}^{M} r_i^2 = \frac{M}{L} S_r^2 + \frac{L}{M}

  where :math:`S_r^2` is the variance (divisor :math:`M`) of the exposure rates
  (:cite:t:`Chen1999`, equations 11 and 14). It is :math:`L/M` when every item is equally exposed,
  and 1 when every examinee takes the same test; higher values mean examinees share more items,
  which weakens test security. It is the overlap rate of :cite:t:`Bar10`. The average over the
  :math:`N(N-1)/2` pairs of distinct examinees is :math:`(N T - 1) / (N - 1)` (:cite:t:`Chen1999`,
  equation 10), which :math:`T` exceeds by :math:`(1 - T) / (N - 1)`.

  The Monte Carlo standard error is the delta method's. :math:`T` is a function of the rates, the
  means over the sessions of each session's item indicators :math:`x_j`, and its gradient in them is
  :math:`2 r / L`, so

  .. math::

     \widehat{\mathrm{SE}}(T) = \frac{2}{L} \sqrt{\frac{1}{N(N-1)} \sum_{j=1}^{N} (y_j - \bar{y})^2},
     \qquad y_j = r \cdot x_j = \sum_{i \in t_j} r_i

  where :math:`y_j` sums the exposure rates of the items of session :math:`j`'s test :math:`t_j`;
  :math:`y_j / L` is that test's average overlap with the run's tests, and :math:`T` their mean.
  :cite:t:`Choe2019` prove that the average test overlap rate of fixed-length tests is
  asymptotically normal.

  Parameters
  ----------
  administered_items : array-like
      The run's tests, one row per session holding the indices of the items it administered, so
      every test has the same length. Each row gives an item at most once.
  n_items : int
      The number of items in the bank, :math:`M`.

  Returns
  -------
  Metric
      The overlap rate and its Monte Carlo standard error, NaN for a single session.

  Raises
  ------
  TypeError
      If ``administered_items`` is not a two-dimensional array of integers, or ``n_items`` is not an
      integer.
  ValueError
      If there is no session or the tests have no item, an index is outside the bank, or a session
      gives an item twice.

  Examples
  --------
  Half the examinees take items 0 and 1 and half items 0 and 2, so a pair shares both items with
  probability 1/2 and one item otherwise, and the rates are 1, 0.5, 0.5 and 0:

  >>> from catsim.metrics import overlap_rate
  >>> overlap_rate([[0, 1], [0, 2]], n_items=4).value
  0.75
  """
  tests = _test_matrix(administered_items, n_items)
  n_sessions, test_size = tests.shape
  rates = numpy.bincount(tests.ravel(), minlength=n_items) / n_sessions
  value = _overlap(rates, test_size)
  if n_sessions < 2:  # noqa: PLR2004 - a spread needs two sessions
    return Metric(value, math.nan)
  summed_rates = rates[tests].sum(axis=1)
  return Metric(value, 2.0 / test_size * math.sqrt(float(numpy.var(summed_rates, ddof=1)) / n_sessions))
