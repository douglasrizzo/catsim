"""Simulated CAT runs on an engine, and their aggregate results."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy

from . import cat
from .engine import RunContext
from .irt import icc
from .state import ExposureTracker

if TYPE_CHECKING:
  import numpy.typing as npt

  from .engine import CatEngine
  from .item_bank import ItemBank
  from .state import CatSessionSnapshot, CatSessionState
  from .typing import FloatArray, IntArray


class SimulatedResponseProvider:
  """Draw a simulated examinee's responses from their true ability.

  The response to an item is correct with the probability the item's 4PL model gives at
  ``session.true_theta``, decided by one uniform draw from the run's generator.
  """

  def answer(  # noqa: PLR6301
    self, item_bank: ItemBank, item_id: int, session: CatSessionSnapshot, rng: numpy.random.Generator
  ) -> bool:
    """Return a correct response with the item's probability of a correct answer at the true ability.

    Parameters
    ----------
    item_bank : ItemBank
        The engine's item bank.
    item_id : int
        Index of the item the selector chose.
    session : CatSessionSnapshot
        The session as it stands before the response; it must have a true ability.
    rng : numpy.random.Generator
        The run's random generator.

    Returns
    -------
    bool
        True for a correct response.

    Raises
    ------
    ValueError
        If the session has no true ability.
    """
    if session.true_theta is None:
      msg = "true_theta is required to simulate responses"
      raise ValueError(msg)

    item = item_bank.get_item(item_id)
    probability = icc(session.true_theta, item[0], item[1], item[2], item[3])
    return bool(probability >= rng.uniform())


def _true_thetas(true_thetas: npt.ArrayLike) -> FloatArray:
  """Return the true abilities as a float array, checking there is one finite number per examinee.

  Parameters
  ----------
  true_thetas : array-like
      The true abilities passed to :meth:`SimulationRunner.run`.

  Returns
  -------
  numpy.ndarray
      The abilities as a one-dimensional float array.

  Raises
  ------
  TypeError
      If ``true_thetas`` is a single number or has more than one dimension.
  ValueError
      If ``true_thetas`` is empty or holds a value that is not finite.
  """
  thetas = numpy.asarray(true_thetas, dtype=float)
  if thetas.ndim == 0:
    msg = (
      f"true_thetas must hold one true ability per examinee, not a single number ({true_thetas!r}); "
      "to simulate n examinees, pass their abilities, e.g. rng.normal(0, 1, n)"
    )
    raise TypeError(msg)
  if thetas.ndim != 1:
    msg = f"true_thetas must be one-dimensional, one true ability per examinee, got shape {thetas.shape}"
    raise TypeError(msg)
  if thetas.size == 0:
    msg = "true_thetas cannot be empty"
    raise ValueError(msg)
  if not numpy.isfinite(thetas).all():
    msg = "true_thetas must be finite numbers"
    raise ValueError(msg)
  return thetas


@dataclass(slots=True)
class SimulationResult:
  """The sessions of a simulated run, with the run's item exposure and duration.

  :meth:`SimulationRunner.run` returns one; one can also be built from sessions an application ran
  itself, whose true abilities may be unknown. ``exposure_counts`` holds each item's
  administrations over the run and ``duration`` the run's wall-clock time in seconds.
  """

  item_bank: ItemBank
  sessions: list[CatSessionState]
  exposure_counts: IntArray
  duration: float

  @property
  def exposure_rates(self) -> FloatArray:
    """Return per-item exposure rates across the simulated sessions."""
    n_sessions = len(self.sessions)
    if n_sessions == 0:
      return numpy.zeros(self.item_bank.n_items, dtype=float)
    return self.exposure_counts.astype(float) / n_sessions

  @property
  def administered_items(self) -> list[list[int]]:
    """Return administered item ids for each session."""
    return [session.administered_item_ids for session in self.sessions]

  @property
  def theta_histories(self) -> list[list[float]]:
    """Return each session's ability estimates, the initial one first."""
    return [list(session.theta_history) for session in self.sessions]

  @property
  def response_vectors(self) -> list[list[bool]]:
    """Return response vectors for each session."""
    return [session.responses for session in self.sessions]

  @property
  def final_thetas(self) -> list[float]:
    """Return each session's final ability estimate."""
    return [session.current_theta for session in self.sessions]

  @property
  def final_ses(self) -> list[float]:
    """Return the standard error of each session's final ability estimate.

    It is infinite for a session that ended before its first response.
    """
    return [session.current_se for session in self.sessions]

  @property
  def test_lengths(self) -> list[int]:
    """Return how many items each session administered."""
    return [session.administered_count for session in self.sessions]

  @property
  def true_thetas(self) -> FloatArray:
    """Return each session's true ability, NaN for a session that has none."""
    return numpy.array(
      [numpy.nan if session.true_theta is None else session.true_theta for session in self.sessions], dtype=float
    )

  @property
  def evaluations(self) -> list[int | None]:
    """Return how many objective evaluations each session's estimates took, summed over its responses.

    The initial estimate, which no estimator computed, counts 0. A session is None when the estimator
    gave no count for one of its estimates (:attr:`~catsim.context.Estimate.evaluations`).
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
    """Return the true abilities, raising if a session has none.

    Returns
    -------
    numpy.ndarray
        Each session's true ability.

    Raises
    ------
    ValueError
        If the result has no session, or any session has no true ability.
    """
    if not self.sessions:
      msg = "bias, mse and rmse need at least one session, and the result has none"
      raise ValueError(msg)
    true_thetas = self.true_thetas
    missing = int(numpy.isnan(true_thetas).sum())
    if missing:
      msg = (
        f"bias, mse and rmse need every session's true theta, but {missing} of {len(self.sessions)} sessions have none"
      )
      raise ValueError(msg)
    return true_thetas

  @property
  def bias(self) -> float:
    """Return the mean of the final estimates minus the true abilities.

    Raises
    ------
    ValueError
        If the result has no session, or any session has no true ability.
    """
    return cat.bias(self._known_true_thetas(), self.final_thetas)

  @property
  def mse(self) -> float:
    """Return the mean squared difference between the final estimates and the true abilities.

    Raises
    ------
    ValueError
        If the result has no session, or any session has no true ability.
    """
    return cat.mse(self._known_true_thetas(), self.final_thetas)

  @property
  def rmse(self) -> float:
    """Return the square root of :attr:`mse`.

    Raises
    ------
    ValueError
        If the result has no session, or any session has no true ability.
    """
    return cat.rmse(self._known_true_thetas(), self.final_thetas)

  @property
  def overlap_rate(self) -> float | None:
    """Return the test overlap rate, or None unless every session administered the same number of items.

    The overlap-rate formula (:func:`catsim.cat.overlap_rate`) assumes one fixed test length, so
    there is no overlap rate for a run of variable-length tests, for a run with no sessions or for
    tests of no items.
    """
    if not self.sessions:
      return None
    test_sizes = set(self.test_lengths)
    if len(test_sizes) != 1:
      return None
    test_size = test_sizes.pop()
    if test_size == 0:
      return None
    return cat.overlap_rate(self.exposure_rates, test_size)


class SimulationRunner:
  """Run a simulated CAT session on an engine for each of a set of examinees.

  Parameters
  ----------
  engine : CatEngine
      The engine whose item bank and strategies every simulated session uses.
  """

  def __init__(self, engine: CatEngine) -> None:
    self._engine = engine

  @property
  def engine(self) -> CatEngine:
    """Return the engine the runner simulates sessions on."""
    return self._engine

  def run(
    self,
    true_thetas: npt.ArrayLike,
    *,
    rng: int | numpy.random.Generator = 0,
    progress: bool = False,
  ) -> SimulationResult:
    """Run one simulated session per true ability, one after another.

    One random generator serves the whole run: the initializer, the selectors' tie-breaking and
    randomization and the simulated responses of every session draw from it, in session order
    (ADR-0013). It is built from ``rng`` at the start of every call, so the same seed gives the same
    result however many times the runner has run.

    Parameters
    ----------
    true_thetas : array-like
        True abilities of the simulated examinees: a non-empty, one-dimensional sequence of finite
        numbers. Session ``i`` simulates the examinee of ability ``true_thetas[i]`` and has session
        id ``i``. To simulate ``n`` examinees from a population, draw their abilities, e.g.
        ``numpy.random.default_rng(1).normal(0, 1, n)``.
    rng : int or numpy.random.Generator, optional
        Seed of the run's random generator, or a generator to draw from, which the run advances, so
        passing the same generator to another run gives other results.
        Default is 0, so a call that passes no seed is deterministic.
    progress : bool, optional
        Whether to show a ``tqdm`` progress bar over the sessions. Default is False.

    Returns
    -------
    SimulationResult
        The sessions, in the order of ``true_thetas``, with the run's exposure counts and duration.

    Raises
    ------
    TypeError
        If ``true_thetas`` is a single number or has more than one dimension.
    ValueError
        If ``true_thetas`` is empty or holds a value that is not finite.
    """
    thetas = _true_thetas(true_thetas)
    tracker = ExposureTracker(self._engine.item_bank.n_items)
    context = RunContext(rng=numpy.random.default_rng(rng), exposure_tracker=tracker)
    response_provider = SimulatedResponseProvider()
    abilities = thetas.tolist()
    if progress:
      # A progress bar is rarely asked for, so importing catsim does not pay for tqdm's import.
      from tqdm import tqdm  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

      abilities = tqdm(abilities)

    start = time.perf_counter()
    sessions = []
    for session_id, true_theta in enumerate(abilities):
      session = self._engine.start_session(session_id=session_id, context=context, true_theta=true_theta)
      sessions.append(self._engine.run_session(session, response_provider, context))
    duration = time.perf_counter() - start

    return SimulationResult(
      item_bank=self._engine.item_bank,
      sessions=sessions,
      exposure_counts=tracker.counts.copy(),
      duration=duration,
    )
