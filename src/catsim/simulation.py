"""Simulated CAT runs on an engine, and their aggregate results."""

from __future__ import annotations

import enum
import secrets
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy

from . import metrics
from .engine import CatEngine

if TYPE_CHECKING:
  from collections.abc import Callable, Iterator

  import numpy.typing as npt

  from .item_bank import ItemBank
  from .session import Session
  from .typing import FloatArray, IntArray


class _Stream(enum.IntEnum):
  """The last entry of an examinee's spawn key, which says which of its two random streams it is (ADR-0013)."""

  PATTERN = 0  # the examinee's response pattern, drawn before its test
  SESSION = 1  # the generator of its session's steps


def simulate(
  engine: CatEngine,
  *,
  true_thetas: npt.ArrayLike | None = None,
  responses: npt.ArrayLike | None = None,
  rng: int | numpy.random.Generator | None = None,
  progress: Callable[[int, int], object] | None = None,
) -> SimulationResult:
  r"""Run one CAT session per examinee on an engine, one after another.

  An examinee is a response pattern, one answer per item of the bank, and its session takes the
  answer to every item it is given from that pattern (ADR-0014). The examinees are given in one of
  three ways:

  - ``true_thetas`` alone: generated examinees. Before its test, examinee ``i`` draws one uniform
    number :math:`u_j` per item of the bank, and its answer to item :math:`j` is correct when
    :math:`u_j < P_j(\theta_i)`, the item's probability of a correct answer at the true ability
    :math:`\theta_i` (``true_thetas[i]``).
  - ``responses`` alone: recorded examinees, whose true abilities are unknown. Row ``i`` holds
    examinee ``i``'s answers. The items whose answer is missing are excluded from the start of the
    session (ADR-0011), so a recorded examinee is given only items they answered, and one who runs
    out of them ends with ``StopReason.NO_ITEMS``.
  - Both: recorded examinees whose true abilities are known, as when a study draws its own response
    matrix with a response model of its choice.

  Every random draw comes from ``rng`` (ADR-0013). The run takes one root entropy from it, and
  examinee ``i``, at position ``i`` of the run, draws from two streams of its own:
  ``SeedSequence(entropy, spawn_key=(i, 0))`` for its response pattern, and
  ``SeedSequence(entropy, spawn_key=(i, 1))`` for its session, the generator of its ``start``,
  ``select`` and ``answer`` steps, from which the initializer, the selector and the exposure control
  draw. A generated examinee therefore gives the same answer to an item whatever the engine and
  whenever the item comes, so two engines run on the same seed and examinees are compared on common
  random numbers.

  Each session's selections are capped against catR's exposure rates: each item's administrations
  over the sessions before it (ADR-0012), so the first session sees no rates. This exposure feedback
  is the only way one session depends on another: without an ``r_max`` cap, an examinee's session
  does not depend on the other examinees.

  Parameters
  ----------
  engine : CatEngine
      The engine whose item bank and strategies every session uses.
  true_thetas : array-like or None, optional
      The examinees' true abilities: a non-empty, one-dimensional sequence of finite numbers. To
      simulate ``n`` examinees from a population, draw their abilities, e.g.
      ``numpy.random.default_rng(1).normal(0, 1, n)``. Default is None, for recorded examinees whose
      abilities are unknown.
  responses : array-like or None, optional
      The recorded examinees' answers: a matrix with one row per examinee and one column per item of
      the bank, of Booleans, of the integers 0 and 1, or of the floats 0.0 and 1.0 with NaN where an
      answer is missing. Default is None, for generated examinees.
  rng : int, numpy.random.Generator or None, optional
      The source of the run's root entropy. None, the default, takes fresh entropy, so every call
      gives other tests. A non-negative int is the entropy itself, so the same seed gives the same
      result. From a generator, one 128-bit number is drawn as the entropy, which advances it, so
      passing the same generator to another run gives other results; a call whose inputs fail
      their checks draws nothing from it.
  progress : callable or None, optional
      Called as ``progress(finished, total)`` after each session ends, with the number of sessions
      finished and the number of examinees, such as ``lambda finished, total: bar.update(1)`` for a
      progress bar ``bar``. Default is None.

  Returns
  -------
  SimulationResult
      The sessions, in examinee order, with the true abilities (None for recorded examinees whose
      abilities are unknown), the run's exposure and proposal counts and its duration.

  Raises
  ------
  TypeError
      If ``engine`` is not a :class:`~catsim.CatEngine`; neither ``true_thetas`` nor ``responses``
      is given; ``true_thetas`` is a single number, holds values that are not real numbers or has
      more than one dimension; ``responses`` is not two-dimensional or holds values that are neither
      Booleans nor real numbers; ``rng`` is neither None, an integer nor a generator; or ``progress``
      is neither None nor callable.
  ValueError
      If ``true_thetas`` or ``responses`` is empty; a true ability is not finite; ``responses`` does
      not have one column per item of the bank, or holds a response other than 0, 1 and NaN; the two
      inputs have different numbers of examinees; or ``rng`` is a negative integer.

  Examples
  --------
  Generated examinees, drawn from a standard normal population:

  >>> import numpy
  >>> from catsim import CatEngine, ItemBank, simulate
  >>> from catsim.estimation import EAPEstimator
  >>> from catsim.initialization import FixedPointInitializer
  >>> from catsim.selection import MaxInfoSelector
  >>> from catsim.stopping import MaxLengthStopper
  >>> bank = ItemBank.generate(50, rng=0)
  >>> engine = CatEngine(bank, FixedPointInitializer(0.0), MaxInfoSelector(), EAPEstimator(), MaxLengthStopper(10))
  >>> result = simulate(engine, true_thetas=numpy.random.default_rng(1).normal(0, 1, 100), rng=0)
  >>> len(result.sessions), result.test_lengths.tolist() == [10] * 100
  (100, True)

  Two recorded examinees: the first answered every item correctly, the second only items 3, 8 and
  15, so the second runs out of answered items after three:

  >>> responses = numpy.full((2, bank.n_items), numpy.nan)
  >>> responses[0] = 1.0
  >>> responses[1, [3, 8, 15]] = [1.0, 0.0, 1.0]
  >>> result = simulate(engine, responses=responses, rng=0)
  >>> [(len(session.administered_items), str(session.stop_reason)) for session in result.sessions]
  [(10, 'stopper'), (3, 'no_items')]
  >>> sorted(result.sessions[1].administered_items)
  [3, 8, 15]
  >>> result.rmse  # doctest: +ELLIPSIS
  Traceback (most recent call last):
  ...
  ValueError: bias, mse and rmse need the examinees' true abilities, and this result has none; ...
  """
  if not isinstance(engine, CatEngine):
    msg = f"engine is a {type(engine).__name__}; it must be a CatEngine"
    raise TypeError(msg)
  if progress is not None and not callable(progress):
    msg = f"progress is {progress!r}; it must be None or a callable, called as progress(finished, total)"
    raise TypeError(msg)
  thetas, matrix, n_examinees = _examinees(true_thetas, responses, engine.item_bank.n_items)
  # Last, so that a call that fails its checks does not advance a generator passed as rng.
  entropy = _entropy(rng)

  exposure_counts = numpy.zeros(engine.item_bank.n_items, dtype=int)
  proposal_counts = numpy.zeros(engine.item_bank.n_items, dtype=int)
  sessions: list[Session] = []
  start = time.perf_counter()
  for i, row in enumerate(_rows(engine.item_bank, thetas, matrix, entropy)):
    # catR's rate: each item's administrations over the sessions started before this one, so the
    # first session sees none (ADR-0012). Only this session changes the counts until it ends, and
    # it is never offered an item again once given it, so its rates are fixed when it starts.
    rates = exposure_counts / i if i else None
    if rates is not None:
      rates.flags.writeable = False
    session, declined = _run_session(engine, row, _stream(entropy, i, _Stream.SESSION), rates)
    for item in session.administered_items:
      exposure_counts[item] += 1
    for item in (*session.administered_items, *declined):
      proposal_counts[item] += 1
    sessions.append(session)
    if progress is not None:
      progress(i + 1, n_examinees)
  duration = time.perf_counter() - start

  return SimulationResult(
    item_bank=engine.item_bank,
    sessions=sessions,
    exposure_counts=exposure_counts,
    duration=duration,
    proposal_counts=proposal_counts,
    true_thetas=thetas,
  )


def _examinees(
  true_thetas: npt.ArrayLike | None, responses: npt.ArrayLike | None, n_items: int
) -> tuple[FloatArray | None, FloatArray | None, int]:
  """Return a run's checked true abilities and recorded responses, either None when not given, and how many examinees.

  Parameters
  ----------
  true_thetas : array-like or None
      The true abilities passed to :func:`simulate`.
  responses : array-like or None
      The recorded responses passed to :func:`simulate`.
  n_items : int
      The number of items in the engine's bank.

  Returns
  -------
  tuple[numpy.ndarray or None, numpy.ndarray or None, int]
      The true abilities as a float array, the responses as a float matrix with NaN where an answer
      is missing, and the number of examinees.

  Raises
  ------
  TypeError
      If neither input is given.
  ValueError
      If both are given, with different numbers of examinees.
  """
  matrix = None if responses is None else _response_matrix(responses, n_items)
  if true_thetas is None:
    if matrix is None:
      msg = "simulate needs examinees: true_thetas for generated ones, responses for recorded ones, or both"
      raise TypeError(msg)
    return None, matrix, len(matrix)
  thetas = _true_thetas(true_thetas)
  if matrix is not None and len(thetas) != len(matrix):
    msg = (
      f"true_thetas has {len(thetas)} entries and responses {len(matrix)} rows; with both, there must be one "
      "true ability per row of responses"
    )
    raise ValueError(msg)
  return thetas, matrix, len(thetas)


def _true_thetas(true_thetas: npt.ArrayLike) -> FloatArray:
  """Return the true abilities as a float array, checking there is one finite number per examinee.

  Parameters
  ----------
  true_thetas : array-like
      The true abilities passed to :func:`simulate`.

  Returns
  -------
  numpy.ndarray
      The abilities as a one-dimensional float array.

  Raises
  ------
  TypeError
      If ``true_thetas`` is a single number, holds values that are not real numbers or has more than
      one dimension.
  ValueError
      If ``true_thetas`` is empty or holds a value that is not finite.
  """
  thetas = numpy.asarray(true_thetas)
  if thetas.ndim == 0:
    msg = (
      f"true_thetas must hold one true ability per examinee, not a single number ({true_thetas!r}); "
      "to simulate n examinees, pass their abilities, e.g. rng.normal(0, 1, n)"
    )
    raise TypeError(msg)
  # Booleans would be read as abilities 0 and 1, and None as NaN.
  if thetas.size and thetas.dtype.kind not in "iuf":
    msg = f"true_thetas holds {thetas.dtype} values; true abilities must be real numbers"
    raise TypeError(msg)
  if thetas.ndim != 1:
    msg = f"true_thetas must be one-dimensional, one true ability per examinee, got shape {thetas.shape}"
    raise TypeError(msg)
  if thetas.size == 0:
    msg = "true_thetas cannot be empty"
    raise ValueError(msg)
  thetas = thetas.astype(float)
  # The probability of a correct answer at NaN is NaN, so every generated answer would be wrong.
  if not numpy.isfinite(thetas).all():
    position = int(numpy.flatnonzero(~numpy.isfinite(thetas))[0])
    msg = f"true_thetas[{position}] is {thetas[position].item()!r}; true abilities must be finite numbers"
    raise ValueError(msg)
  return thetas


def _response_matrix(responses: npt.ArrayLike, n_items: int) -> FloatArray:
  """Return recorded responses as a float matrix of 1.0 (correct), 0.0 (incorrect) and NaN (missing), checking them.

  Parameters
  ----------
  responses : array-like
      The recorded responses passed to :func:`simulate`.
  n_items : int
      The number of items in the engine's bank.

  Returns
  -------
  numpy.ndarray
      A copy of the responses, one row per examinee and one column per item.

  Raises
  ------
  TypeError
      If the responses are not two-dimensional, or hold values that are neither Booleans nor real
      numbers.
  ValueError
      If there is no row, not one column per item, or a response other than 0, 1 and, among floats,
      NaN.
  """
  matrix = numpy.asarray(responses)
  # Before the shape check: an empty list has one dimension, and is empty like true_thetas=[].
  if matrix.ndim in {1, 2} and len(matrix) == 0:
    msg = "responses cannot be empty"
    raise ValueError(msg)
  if matrix.ndim != 2:  # noqa: PLR2004 - a matrix
    msg = (
      "responses must be two-dimensional, one row per examinee and one column per item of the bank, got shape "
      f"{matrix.shape}"
    )
    raise TypeError(msg)
  if matrix.shape[1] != n_items:
    msg = f"responses has shape {matrix.shape}; there must be one column per item of the bank ({n_items})"
    raise ValueError(msg)
  kind = matrix.dtype.kind
  if kind == "b":
    return matrix.astype(float)
  # Strings and objects are no responses: None would mark a missing answer that NaN marks.
  if kind not in "iuf":
    msg = (
      f"responses holds {matrix.dtype} values; responses must be bools, the integers 0 and 1, or the floats 0.0 "
      "and 1.0 with NaN for a missing response"
    )
    raise TypeError(msg)
  valid = (matrix == 0) | (matrix == 1)
  if kind == "f":
    valid |= numpy.isnan(matrix)
  if not valid.all():
    row, column = (int(index) for index in numpy.argwhere(~valid)[0])
    rule = (
      "an integer response must be 0 or 1" if kind in "iu" else "a response must be 0 or 1, or NaN where it is missing"
    )
    msg = f"responses[{row}, {column}] is {matrix[row, column].item()!r}; {rule}"
    raise ValueError(msg)
  return matrix.astype(float)


def _entropy(rng: object) -> int:
  """Return a run's root entropy: the seed itself, a number drawn once from a generator, or fresh entropy (ADR-0013).

  Parameters
  ----------
  rng : object
      The ``rng`` passed to :func:`simulate`.

  Returns
  -------
  int
      The root entropy of the examinees' seed sequences.

  Raises
  ------
  TypeError
      If ``rng`` is neither None, an integer nor a generator.
  ValueError
      If ``rng`` is a negative integer.
  """
  if rng is None:
    # The size of SeedSequence's own fresh entropy.
    return secrets.randbits(128)
  if isinstance(rng, numpy.random.Generator):
    return int.from_bytes(rng.bytes(16), "little")
  # A bool is an int, but True passed as a seed is a flag passed by mistake.
  if isinstance(rng, (bool, numpy.bool_)) or not isinstance(rng, (int, numpy.integer)):
    msg = f"rng is {rng!r}; it must be None, an integer seed or a numpy.random.Generator"
    raise TypeError(msg)
  if rng < 0:
    msg = f"rng is {rng}; a seed must not be negative"
    raise ValueError(msg)
  return int(rng)


def _stream(entropy: int, examinee: int, stream: _Stream) -> numpy.random.Generator:
  """Return the generator of one of an examinee's two random streams (ADR-0013).

  Parameters
  ----------
  entropy : int
      The run's root entropy.
  examinee : int
      The examinee's position in the run.
  stream : _Stream
      Which of the examinee's streams.

  Returns
  -------
  numpy.random.Generator
      A generator seeded by ``SeedSequence(entropy, spawn_key=(examinee, stream))``.
  """
  return numpy.random.default_rng(numpy.random.SeedSequence(entropy, spawn_key=(examinee, stream)))


def _rows(
  item_bank: ItemBank, true_thetas: FloatArray | None, responses: FloatArray | None, entropy: int
) -> Iterator[FloatArray]:
  """Yield each examinee's answers to every item, in examinee order: the recorded row, or one drawn before the test.

  A generated examinee draws one uniform number per item from its pattern stream, and its answer to
  an item is correct when the number falls below the item's probability at its true ability
  (ADR-0014).

  Parameters
  ----------
  item_bank : ItemBank
      The engine's item bank.
  true_thetas : numpy.ndarray or None
      The examinees' true abilities, or None for recorded examinees whose abilities are unknown.
  responses : numpy.ndarray or None
      The recorded responses, or None for generated examinees.
  entropy : int
      The run's root entropy.

  Yields
  ------
  numpy.ndarray
      The examinee's row: 1.0 for a correct answer, 0.0 for an incorrect one and NaN for a missing
      one.
  """
  if responses is not None:
    yield from responses
  elif true_thetas is not None:
    for i, true_theta in enumerate(true_thetas.tolist()):
      draws = _stream(entropy, i, _Stream.PATTERN).random(item_bank.n_items)
      yield (draws < item_bank.icc(true_theta)).astype(float)


def _run_session(
  engine: CatEngine, row: FloatArray, rng: numpy.random.Generator, exposure_rates: FloatArray | None
) -> tuple[Session, list[int]]:
  """Run one session to its end, answering from ``row``, and return it with the items the exposure control declined.

  The items whose answer is missing are excluded from the start. :func:`simulate` builds every
  input of the engine's steps itself, so it takes the engine's unchecked path (ADR-0009).

  Parameters
  ----------
  engine : CatEngine
      The engine.
  row : numpy.ndarray
      The examinee's answers to every item: 1.0 for correct, 0.0 for incorrect and NaN for missing.
  rng : numpy.random.Generator
      The generator of the session's steps.
  exposure_rates : numpy.ndarray or None
      The exposure rates the session's selections are capped against.

  Returns
  -------
  tuple[Session, list[int]]
      The finished session, and the items the exposure control declined in it, in order.
  """
  session = engine._start(rng, tuple(numpy.flatnonzero(numpy.isnan(row)).tolist()))  # noqa: SLF001
  declined: list[int] = []
  while True:
    selection = engine._select(session, rng, exposure_rates)  # noqa: SLF001
    session = selection.session
    declined.extend(selection.declined)
    if selection.item is None:
      return session, declined
    # The row holds 1.0 or 0.0 for every item the session may be given; its NaNs are excluded.
    session = engine._answer(session, selection.item, bool(row[selection.item]), rng)  # noqa: SLF001


@dataclass(slots=True)
class SimulationResult:
  """The sessions of a simulated run, with the run's item exposure and duration.

  :func:`simulate` returns one; one can also be built from sessions an application ran itself,
  whose true abilities may be unknown.

  Parameters
  ----------
  item_bank : ItemBank
      The item bank the sessions drew from.
  sessions : list of Session
      The run's sessions, in the order they ran.
  exposure_counts : numpy.ndarray
      Each item's administrations over the run.
  duration : float
      The run's wall-clock time, in seconds.
  proposal_counts : numpy.ndarray or None, optional
      Each item's proposals: its administrations plus the times the engine's exposure control
      declined it. Default is None, for a result built without them.
  true_thetas : array-like or None, optional
      Each session's true ability, a finite number, in session order, or None when the true
      abilities are unknown, as for recorded examinees. A session does not hold its examinee's true
      ability, so the result does. Default is None.

  Raises
  ------
  ValueError
      If ``true_thetas`` does not hold one finite number per session.
  """

  item_bank: ItemBank
  sessions: list[Session]
  exposure_counts: IntArray
  duration: float
  proposal_counts: IntArray | None = None
  true_thetas: FloatArray | None = None

  def __post_init__(self) -> None:
    """Store the true abilities as a float array, checking that there is one finite number per session."""
    if self.true_thetas is None:
      return
    self.true_thetas = numpy.asarray(self.true_thetas, dtype=float)
    if self.true_thetas.shape != (len(self.sessions),):
      msg = (
        f"true_thetas has shape {self.true_thetas.shape}; there must be one true ability per session "
        f"({len(self.sessions)})"
      )
      raise ValueError(msg)
    # The true abilities are known for every session or for none (ADR-0016): NaN does not mark an unknown one.
    if not numpy.isfinite(self.true_thetas).all():
      position = int(numpy.flatnonzero(~numpy.isfinite(self.true_thetas))[0])
      msg = (
        f"true_thetas[{position}] is {self.true_thetas[position].item()!r}; true abilities must be finite, and "
        "true_thetas None when they are unknown"
      )
      raise ValueError(msg)

  @property
  def exposure_rates(self) -> FloatArray:
    """Return per-item exposure rates across the simulated sessions."""
    n_sessions = len(self.sessions)
    if n_sessions == 0:
      return numpy.zeros(self.item_bank.n_items, dtype=float)
    return self.exposure_counts.astype(float) / n_sessions

  @property
  def proposal_rates(self) -> FloatArray | None:
    """Return the share of the sessions in which the selector proposed each item, or None without proposal counts.

    A session proposes an item at most once, as the exposure control declines it or the session administers it, so
    the rate is at most 1. Without an exposure control it equals :attr:`exposure_rates`.
    """
    if self.proposal_counts is None:
      return None
    n_sessions = len(self.sessions)
    if n_sessions == 0:
      return numpy.zeros(self.item_bank.n_items, dtype=float)
    return self.proposal_counts.astype(float) / n_sessions

  @property
  def administered_items(self) -> list[tuple[int, ...]]:
    """Return each session's administered items, in the order given."""
    return [session.administered_items for session in self.sessions]

  @property
  def theta_histories(self) -> list[list[float]]:
    """Return each session's ability estimates, the initial one first."""
    return [list(session.theta_history) for session in self.sessions]

  @property
  def responses(self) -> list[tuple[bool, ...]]:
    """Return each session's responses, in the order given."""
    return [session.responses for session in self.sessions]

  @property
  def final_thetas(self) -> FloatArray:
    """Return each session's final ability estimate."""
    return numpy.array([session.current_theta for session in self.sessions], dtype=float)

  @property
  def final_ses(self) -> FloatArray:
    """Return the standard error of each session's final ability estimate.

    It is infinite for a session that ended before its first response.
    """
    return numpy.array([session.current_se for session in self.sessions], dtype=float)

  @property
  def test_lengths(self) -> IntArray:
    """Return how many items each session administered."""
    return numpy.array([len(session.administered_items) for session in self.sessions], dtype=int)

  @property
  def evaluations(self) -> list[int | None]:
    """Return how many objective evaluations each session's estimates took, summed over its responses.

    The initial estimate, which no estimator computed, counts 0. A session is None when the estimator
    gave no count for one of its estimates (the ``evaluations`` of an :class:`~catsim.Estimate`).
    """
    totals: list[int | None] = []
    for session in self.sessions:
      counts = [estimate.evaluations for estimate in session.estimates[1:]]
      totals.append(None if None in counts else sum(count for count in counts if count is not None))
    return totals

  @property
  def mean_evaluations(self) -> float | None:
    """Return the mean number of objective evaluations per estimate over the run's responses.

    Only the estimates that have a count take part. None when no estimate of the run has one, as for
    a run whose estimator gives no counts or whose sessions have no responses.
    """
    counts = [
      estimate.evaluations
      for session in self.sessions
      for estimate in session.estimates[1:]
      if estimate.evaluations is not None
    ]
    return float(numpy.mean(counts)) if counts else None

  def _known_true_thetas(self) -> FloatArray:
    """Return the true abilities, raising if the result has none.

    Returns
    -------
    numpy.ndarray
        Each session's true ability.

    Raises
    ------
    ValueError
        If the result has no session, or its true abilities are unknown.
    """
    if not self.sessions:
      msg = "bias, mse and rmse need at least one session, and the result has none"
      raise ValueError(msg)
    if self.true_thetas is None:
      msg = (
        "bias, mse and rmse need the examinees' true abilities, and this result has none; pass true_thetas to "
        "simulate with the responses when they are known"
      )
      raise ValueError(msg)
    return self.true_thetas

  @property
  def bias(self) -> float:
    """Return the mean of the final estimates minus the true abilities, :func:`catsim.metrics.bias`.

    Raises
    ------
    ValueError
        If the result has no session, or its true abilities are unknown.
    """
    return metrics.bias(self._known_true_thetas(), self.final_thetas)

  @property
  def mse(self) -> float:
    """Return the mean squared difference between the final estimates and the true abilities.

    See :func:`catsim.metrics.mse`.

    Raises
    ------
    ValueError
        If the result has no session, or its true abilities are unknown.
    """
    return metrics.mse(self._known_true_thetas(), self.final_thetas)

  @property
  def rmse(self) -> float:
    """Return the square root of :attr:`mse`, :func:`catsim.metrics.rmse`.

    Raises
    ------
    ValueError
        If the result has no session, or its true abilities are unknown.
    """
    return metrics.rmse(self._known_true_thetas(), self.final_thetas)

  @property
  def overlap_rate(self) -> float | None:
    """Return the test overlap rate, or None unless every session administered the same number of items.

    The overlap-rate formula (:func:`catsim.metrics.overlap_rate`) assumes one fixed test length, so
    there is no overlap rate for a run of variable-length tests, for a run with no sessions or for
    tests of no items.
    """
    lengths = self.test_lengths
    if lengths.size == 0:
      return None
    test_size = int(lengths[0])
    if test_size == 0 or numpy.any(lengths != test_size):
      return None
    return metrics.overlap_rate(self.exposure_rates, test_size)
