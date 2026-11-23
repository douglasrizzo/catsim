"""State models for CAT execution and simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy

from . import cat

if TYPE_CHECKING:
  from collections.abc import Hashable

  from ._typing import FloatArray, IntArray
  from .item_bank import ItemBank


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
  int, a str or a UUID. ``theta_history`` starts with the initial estimate and gains one estimate
  per response; its last entry is the current estimate. A session runs while ``stop_reason`` is
  None.

  Raises
  ------
  ValueError
      If the session does not have one response per administered item and one more estimate than
      responses.
  """

  session_id: Hashable
  theta_history: list[float]
  true_theta: float | None = None
  administered_item_ids: list[int] = field(default_factory=list)
  responses: list[bool] = field(default_factory=list)
  stop_reason: StopReason | None = None

  def __post_init__(self) -> None:
    """Reject a session whose items, responses and estimates do not line up."""
    n_responses = len(self.responses)
    if len(self.administered_item_ids) != n_responses:
      msg = (
        f"Session {self.session_id} needs one response per administered item, got "
        f"len(administered_item_ids)={len(self.administered_item_ids)} and len(responses)={n_responses}"
      )
      raise ValueError(msg)
    if len(self.theta_history) != n_responses + 1:
      msg = (
        f"Session {self.session_id} needs the initial estimate and one estimate per response in "
        f"theta_history, got len(theta_history)={len(self.theta_history)} and len(responses)={n_responses}"
      )
      raise ValueError(msg)

  @property
  def current_theta(self) -> float:
    """Return the current ability estimate, the last entry of ``theta_history``."""
    return self.theta_history[-1]

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
      theta_history=tuple(self.theta_history),
      true_theta=self.true_theta,
      administered_item_ids=tuple(self.administered_item_ids),
      responses=tuple(self.responses),
      stop_reason=self.stop_reason,
    )


@dataclass(slots=True, frozen=True)
class CatSessionSnapshot:
  """Immutable, hashable copy of a session at a specific point in time."""

  session_id: Hashable
  theta_history: tuple[float, ...]
  true_theta: float | None
  administered_item_ids: tuple[int, ...]
  responses: tuple[bool, ...]
  stop_reason: StopReason | None

  @property
  def current_theta(self) -> float:
    """Return the ability estimate at the time of the snapshot."""
    return self.theta_history[-1]

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
  """Track item usage counts during a simulation run."""

  n_items: int
  counts: IntArray = field(init=False)

  def __post_init__(self) -> None:
    """Allocate the item exposure counter array."""
    self.counts = numpy.zeros(self.n_items, dtype=int)

  def record(self, item_id: int) -> None:
    """Record one administration for an item."""
    self.counts[item_id] += 1

  def rates(self, n_sessions: int) -> FloatArray:
    """Return exposure rates for the tracked counts."""
    if n_sessions <= 0:
      return numpy.zeros(self.n_items, dtype=float)
    return self.counts.astype(float) / n_sessions


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
    return [session.theta_history for session in self.sessions]

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
