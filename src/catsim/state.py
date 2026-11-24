"""State models for CAT execution and simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy

from . import cat
from .context import Estimate

if TYPE_CHECKING:
  from collections.abc import Hashable

  from .item_bank import ItemBank
  from .typing import FloatArray, IntArray


class StopReason(StrEnum):
  """Why a CAT session ended.

  ``STOPPER`` means the engine's stopper was satisfied by the session's current state. ``NO_ITEMS``
  means the selector had no item to offer, which is not necessarily the end of the item bank: a
  selector that follows a fixed list or works through strata can run out first.
  """

  STOPPER = "stopper"
  NO_ITEMS = "no_items"


@dataclass(slots=True)
class CatSessionState:
  """Mutable runtime state for a single examinee CAT session.

  ``session_id`` is whatever hashable value the application identifies the session by, such as an
  int, a str or a UUID. ``estimates`` starts with the initial estimate, whose standard error is
  ``math.inf`` and which took no evaluation, and gains the :class:`~catsim.context.Estimate` the
  estimator returned for each response; its last entry is the current estimate. A session runs
  while ``stop_reason`` is None.

  Raises
  ------
  ValueError
      If the session does not have one response per administered item and one more estimate than
      responses, or if ``stop_reason`` is not a :class:`StopReason` value.
  TypeError
      If an entry of ``estimates`` is not an :class:`~catsim.context.Estimate`.
  """

  session_id: Hashable
  estimates: list[Estimate]
  true_theta: float | None = None
  administered_item_ids: list[int] = field(default_factory=list)
  responses: list[bool] = field(default_factory=list)
  stop_reason: StopReason | None = None

  def __post_init__(self) -> None:
    """Take its own copy of the histories and reject a session whose entries do not line up."""
    # The engine appends to these lists, so they must be lists the session owns: a snapshot's tuples
    # would fail half-way through a step, and a caller's lists would change under the caller.
    self.administered_item_ids = list(self.administered_item_ids)
    self.responses = list(self.responses)
    self.estimates = list(self.estimates)
    if not all(isinstance(estimate, Estimate) for estimate in self.estimates):
      # A session restored from storage needs its estimates rebuilt, not bare thetas or mappings.
      msg = f"Session {self.session_id} needs Estimate entries in estimates, got {self.estimates!r}"
      raise TypeError(msg)
    if self.stop_reason is not None:
      # A stop reason restored from storage comes back as its string value.
      self.stop_reason = StopReason(self.stop_reason)
    n_responses = len(self.responses)
    if len(self.administered_item_ids) != n_responses:
      msg = (
        f"Session {self.session_id} needs one response per administered item, got "
        f"len(administered_item_ids)={len(self.administered_item_ids)} and len(responses)={n_responses}"
      )
      raise ValueError(msg)
    if len(self.estimates) != n_responses + 1:
      msg = (
        f"Session {self.session_id} needs the initial estimate and one estimate per response in "
        f"estimates, got len(estimates)={len(self.estimates)} and len(responses)={n_responses}"
      )
      raise ValueError(msg)

  @property
  def theta_history(self) -> tuple[float, ...]:
    """Return the ability estimates in order, the initial one first."""
    return tuple(estimate.theta for estimate in self.estimates)

  @property
  def current_theta(self) -> float:
    """Return the current ability estimate, that of the last entry of ``estimates``."""
    return self.estimates[-1].theta

  @property
  def current_se(self) -> float:
    """Return the standard error of the current ability estimate."""
    return self.estimates[-1].se

  @property
  def is_stopped(self) -> bool:
    """Return whether the session has ended."""
    return self.stop_reason is not None

  @property
  def administered_count(self) -> int:
    """Return the number of administered items."""
    return len(self.administered_item_ids)

  def snapshot(self) -> CatSessionSnapshot:
    """Return an immutable copy of the current session state."""
    return CatSessionSnapshot(
      session_id=self.session_id,
      estimates=tuple(self.estimates),
      true_theta=self.true_theta,
      administered_item_ids=tuple(self.administered_item_ids),
      responses=tuple(self.responses),
      stop_reason=self.stop_reason,
    )


@dataclass(slots=True, frozen=True)
class CatSessionSnapshot:
  """Immutable, hashable copy of a session at a specific point in time."""

  session_id: Hashable
  estimates: tuple[Estimate, ...]
  true_theta: float | None
  administered_item_ids: tuple[int, ...]
  responses: tuple[bool, ...]
  stop_reason: StopReason | None

  @property
  def theta_history(self) -> tuple[float, ...]:
    """Return the ability estimates up to the snapshot, the initial one first."""
    return tuple(estimate.theta for estimate in self.estimates)

  @property
  def current_theta(self) -> float:
    """Return the ability estimate at the time of the snapshot."""
    return self.estimates[-1].theta

  @property
  def current_se(self) -> float:
    """Return the standard error of the ability estimate at the time of the snapshot."""
    return self.estimates[-1].se

  @property
  def is_stopped(self) -> bool:
    """Return whether the session had ended at the time of the snapshot."""
    return self.stop_reason is not None


@dataclass(slots=True, frozen=True)
class CatStepResult:
  """Result of one CAT step: the item administered, its response and the session as the step left it.

  ``selected_item_id`` and ``response`` are None for a step that administered nothing because the
  session had ended.
  """

  session: CatSessionSnapshot
  selected_item_id: int | None
  response: bool | None

  @property
  def updated_theta(self) -> float:
    """Return the ability estimate after the step."""
    return self.session.current_theta

  @property
  def stopped(self) -> bool:
    """Return whether the session had ended by the end of the step."""
    return self.session.is_stopped

  @property
  def stop_reason(self) -> StopReason | None:
    """Return why the session ended, or None if it is still running."""
    return self.session.stop_reason


@dataclass(slots=True)
class ExposureTracker:
  """Count item administrations and sessions across a run, and give sessions their exposure rates.

  One tracker is shared by every session of a run or of a live deployment, through
  :class:`~catsim.engine.RunContext`. The engine calls :meth:`start_session` when a session starts and
  :meth:`record` for every administered item. ``counts`` holds each item's administrations so far,
  those of sessions still in progress included, and ``n_sessions`` the number of sessions started.

  A session is selected against each item's administrations divided by the number of other sessions
  started so far (ADR-0003). With sessions run one after another, as a simulation runs them, that
  is the share of the examinees tested before the current one, as in catR's
  ``simulateRespondents``; the first session sees every rate at 0. Sessions that run side by side,
  as in a live deployment, see each other's administrations as they happen.
  """

  n_items: int
  counts: IntArray = field(init=False)
  n_sessions: int = field(init=False, default=0)

  def __post_init__(self) -> None:
    """Allocate the item counts, all zero."""
    self.counts = numpy.zeros(self.n_items, dtype=int)

  def start_session(self) -> None:
    """Count a session that has started."""
    self.n_sessions += 1

  def record(self, item_id: int) -> None:
    """Record one administration for an item."""
    self.counts[item_id] += 1

  def rates(self) -> FloatArray:
    """Return the exposure rates a session in progress is selected against.

    Each item's administrations are divided by the sessions started so far other than the one
    asking, so a session never counts itself. Its own administrations stay in the counts, but only
    for items it has already administered and will not be offered again; for those the rate can
    exceed 1.

    Returns
    -------
    numpy.ndarray
        A new array with one rate per item; all zero while at most one session has started. This is
        not the run's final exposure, which :attr:`SimulationResult.exposure_rates` gives.
    """
    others = self.n_sessions - 1
    if others < 1:
      return numpy.zeros(self.n_items)
    return self.counts / others


@dataclass(slots=True)
class SimulationResult:
  """Aggregate result for a batch of CAT sessions."""

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
  def estimations(self) -> list[list[float]]:
    """Return theta history for each session."""
    return [list(session.theta_history) for session in self.sessions]

  @property
  def response_vectors(self) -> list[list[bool]]:
    """Return response vectors for each session."""
    return [session.responses for session in self.sessions]

  @property
  def latest_estimations(self) -> list[float]:
    """Return final theta estimates for each session."""
    return [session.current_theta for session in self.sessions]

  @property
  def examinees(self) -> FloatArray:
    """Return true theta values for the simulated sessions."""
    return numpy.asarray([0.0 if session.true_theta is None else session.true_theta for session in self.sessions])

  @property
  def bias(self) -> float:
    """Return simulation bias."""
    return cat.bias(self.examinees, self.latest_estimations)

  @property
  def mse(self) -> float:
    """Return simulation MSE."""
    return cat.mse(self.examinees, self.latest_estimations)

  @property
  def rmse(self) -> float:
    """Return simulation RMSE."""
    return cat.rmse(self.examinees, self.latest_estimations)

  @property
  def overlap_rate(self) -> float | None:
    """Return overlap rate when all sessions have the same test length."""
    if not self.sessions:
      return None
    test_sizes = {len(session.administered_item_ids) for session in self.sessions}
    if len(test_sizes) != 1:
      return None
    test_size = test_sizes.pop()
    if test_size == 0:
      return None
    return cat.overlap_rate(self.exposure_rates, test_size)
