"""The one place where the reference cases call catsim's step, session, simulation and metrics API.

The cases call catsim's computations directly where their calling convention is stable: the
functions of ``catsim.irt``, the item bank, and each strategy given a ``StepContext``. Everything
whose calling convention the rescoping changes goes through this module instead: building step
contexts, building an engine and running whole tests, the exposure control the whole tests use, and
the metric functions. A later item that changes one of those rewrites the functions here, keeping
their signatures and what they return, and leaves the cases and the recorded data as they are.

This version is written for catsim at adda72dc: ``CatEngine.start_session`` and ``run_session``
with a ``RunContext``, an ``ExposureTracker`` and a ``ResponseProvider``.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy
from typing_extensions import override

from catsim import CatEngine, ExposureTracker, ResponseProvider, RunContext, SimulationResult, StepContext, metrics
from catsim.exposure import BaseExposureControl

if TYPE_CHECKING:
  from collections.abc import Sequence

  import numpy.typing as npt

  from catsim import CatSessionSnapshot, ItemBank
  from catsim.estimation import BaseEstimator
  from catsim.initialization import BaseInitializer
  from catsim.selection import BaseSelector
  from catsim.stopping import BaseStopper
  from catsim.typing import FloatArray


def step_context(
  item_bank: ItemBank,
  *,
  rng: numpy.random.Generator,
  administered_items: Sequence[int] = (),
  responses: Sequence[bool] = (),
  excluded_items: Sequence[int] = (),
  theta: float = 0.0,
  se: float = math.inf,
  exposure_rates: FloatArray | None = None,
) -> StepContext:
  """Return the context a strategy reads for a session in the given state."""
  return StepContext(
    item_bank=item_bank,
    rng=rng,
    administered_items=tuple(administered_items),
    responses=tuple(responses),
    excluded_items=tuple(excluded_items),
    theta=theta,
    se=se,
    exposure_rates=exposure_rates,
  )


def excluding(context: StepContext, item: int) -> StepContext:
  """Return ``context`` with one more excluded item, as a selector's next proposal would see it."""
  return dataclasses.replace(context, excluded_items=(*context.excluded_items, item))


class DeclineListed(BaseExposureControl):
  """An exposure control that declines every proposal of a listed item, and draws no random number.

  Parameters
  ----------
  items : Sequence[int]
      The items to decline.
  """

  def __init__(self, items: Sequence[int]) -> None:
    self._items = frozenset(items)

  @override
  def accept(self, context: StepContext, item_id: int) -> bool:
    """Return whether the item is not one of the listed ones."""
    return item_id not in self._items


def build_engine(
  item_bank: ItemBank,
  initializer: BaseInitializer,
  selector: BaseSelector,
  estimator: BaseEstimator,
  stopper: BaseStopper,
  *,
  exposure_control: BaseExposureControl | None = None,
) -> CatEngine:
  """Return an engine that tests on ``item_bank`` with the given strategies."""
  return CatEngine(item_bank, initializer, selector, estimator, stopper, exposure_control=exposure_control)


@dataclass(frozen=True, slots=True)
class SessionRecord:
  """What a finished session of a whole test recorded.

  Parameters
  ----------
  administered_items : list[int]
      The items given, in order.
  responses : list[bool]
      Their responses, in the same order.
  thetas : list[float]
      The estimates, the initial one first and then one per response.
  ses : list[float]
      The standard error of each estimate.
  excluded_items : list[int]
      The items the session excluded, in the order they were excluded.
  stop_reason : str
      The value of the session's stop reason.
  final_context : StepContext
      The finished session as its strategies would read it, with a fresh generator and no exposure
      rates.
  """

  administered_items: list[int]
  responses: list[bool]
  thetas: list[float]
  ses: list[float]
  excluded_items: list[int]
  stop_reason: str
  final_context: StepContext


@dataclass(frozen=True, slots=True)
class RecordedRun:
  """The sessions of a run on recorded responses, with the run's exposure and its metrics.

  Parameters
  ----------
  sessions : list[SessionRecord]
      One session per examinee, in examinee order.
  exposure_counts : list[int]
      Each item's administrations over the run.
  proposal_counts : list[int]
      Each item's proposals: its administrations plus the exposure control's declines.
  exposure_rates : list[float]
      The exposure counts divided by the number of sessions.
  proposal_rates : list[float]
      The proposal counts divided by the number of sessions.
  overlap_rate : float or None
      The test overlap rate, or None when the sessions differ in length.
  bias, mse, rmse : float or None
      The accuracy of the final estimates, or None when the true abilities are unknown.
  """

  sessions: list[SessionRecord]
  exposure_counts: list[int]
  proposal_counts: list[int]
  exposure_rates: list[float]
  proposal_rates: list[float]
  overlap_rate: float | None
  bias: float | None
  mse: float | None
  rmse: float | None


class _RecordedExaminee(ResponseProvider):
  """Answer every item from one row of a response matrix, 1.0 for correct and 0.0 for incorrect."""

  def __init__(self, row: FloatArray) -> None:
    self._row = row

  @override
  def answer(self, item_bank: ItemBank, item_id: int, session: CatSessionSnapshot, rng: numpy.random.Generator) -> bool:
    """Return the recorded response to the item."""
    response = float(self._row[item_id])
    if math.isnan(response):
      msg = f"item {item_id} has no recorded response, so the session should have excluded it"
      raise ValueError(msg)
    return response == 1.0


def run_recorded(
  engine: CatEngine,
  responses: npt.ArrayLike,
  *,
  rng: int,
  true_thetas: Sequence[float] | None = None,
) -> RecordedRun:
  """Run one session per row of a response matrix, one after another, with exposure feedback.

  Each examinee answers from their row: 1 for correct, 0 for incorrect and NaN for an item they did
  not answer, which their session excludes from the start (ADR-0014). Each session sees the
  exposure rates of the sessions before it (ADR-0012).

  Parameters
  ----------
  engine : CatEngine
      The engine to run.
  responses : array-like
      One row per examinee and one column per item of the bank.
  rng : int
      Seed of the run's random draws.
  true_thetas : Sequence[float] or None, optional
      Each examinee's true ability, if known. Default is None.

  Returns
  -------
  RecordedRun
      The sessions and the run's exposure and metrics.
  """
  matrix = numpy.asarray(responses, dtype=float)
  tracker = ExposureTracker(engine.item_bank.n_items)
  context = RunContext(rng=numpy.random.default_rng(rng), exposure_tracker=tracker)
  sessions = []
  for index, row in enumerate(matrix):
    true_theta = None if true_thetas is None else float(true_thetas[index])
    state = engine.start_session(index, context, true_theta=true_theta)
    state.excluded_items.extend(numpy.flatnonzero(numpy.isnan(row)).tolist())
    sessions.append(engine.run_session(state, _RecordedExaminee(row), context))
  result = SimulationResult(
    item_bank=engine.item_bank,
    sessions=sessions,
    exposure_counts=tracker.counts.copy(),
    duration=0.0,
    proposal_counts=tracker.proposals.copy(),
  )
  proposal_rates = result.proposal_rates
  assert proposal_rates is not None
  known = true_thetas is not None
  return RecordedRun(
    sessions=[
      SessionRecord(
        administered_items=list(session.administered_items),
        responses=list(session.responses),
        thetas=[estimate.theta for estimate in session.estimates],
        ses=[estimate.se for estimate in session.estimates],
        excluded_items=list(session.excluded_items),
        stop_reason=str(session.stop_reason),
        final_context=engine.context(session),
      )
      for session in sessions
    ],
    exposure_counts=result.exposure_counts.tolist(),
    proposal_counts=tracker.proposals.tolist(),
    exposure_rates=result.exposure_rates.tolist(),
    proposal_rates=proposal_rates.tolist(),
    overlap_rate=result.overlap_rate,
    bias=result.bias if known else None,
    mse=result.mse if known else None,
    rmse=result.rmse if known else None,
  )


def bias(actual: npt.ArrayLike, predicted: npt.ArrayLike) -> float:
  """Return the mean of the estimates minus the true abilities."""
  return metrics.bias(actual, predicted)


def mse(actual: npt.ArrayLike, predicted: npt.ArrayLike) -> float:
  """Return the mean squared difference between the estimates and the true abilities."""
  return metrics.mse(actual, predicted)


def rmse(actual: npt.ArrayLike, predicted: npt.ArrayLike) -> float:
  """Return the square root of the mean squared error."""
  return metrics.rmse(actual, predicted)


def overlap_rate(exposure_rates: FloatArray, test_size: int) -> float:
  """Return the expected share of items two examinees' tests of ``test_size`` items share."""
  return metrics.overlap_rate(exposure_rates, test_size)
