"""Simulation runner built on top of the explicit CAT engine."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy

from .engine import RunContext, SimulatedResponseProvider
from .state import ExposureTracker, SimulationResult

if TYPE_CHECKING:
  import numpy.typing as npt

  from .engine import CatEngine
  from .typing import FloatArray


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
        Seed of the run's random generator, or a generator to draw from, which the run advances.
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
