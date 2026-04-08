"""State models for CAT execution and simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy
import numpy.typing as npt

from . import cat

if TYPE_CHECKING:
  from .item_bank import ItemBank


class SessionStatus(Enum):
  """Lifecycle state for a CAT session."""

  NOT_STARTED = "not_started"
  IN_PROGRESS = "in_progress"
  STOPPED = "stopped"


@dataclass(slots=True)
class CatSessionState:
  """Mutable runtime state for a single examinee CAT session."""

  session_id: int
  current_theta: float
  true_theta: float | None = None
  administered_item_ids: list[int] = field(default_factory=list)
  responses: list[bool] = field(default_factory=list)
  theta_history: list[float] = field(default_factory=list)
  status: SessionStatus = SessionStatus.NOT_STARTED
  stop_reason: str | None = None
  metadata: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    """Seed theta history with the initial theta when no history is provided."""
    if not self.theta_history:
      self.theta_history.append(self.current_theta)

  @property
  def administered_count(self) -> int:
    """Return the number of administered items."""
    return len(self.administered_item_ids)

  @property
  def latest_theta(self) -> float:
    """Return the latest theta estimate."""
    return self.theta_history[-1]

  def snapshot(self) -> CatSessionSnapshot:
    """Return an immutable snapshot of the current session state."""
    return CatSessionSnapshot(
      session_id=self.session_id,
      current_theta=self.current_theta,
      true_theta=self.true_theta,
      administered_item_ids=self.administered_item_ids.copy(),
      responses=self.responses.copy(),
      theta_history=self.theta_history.copy(),
      status=self.status,
      stop_reason=self.stop_reason,
      metadata=self.metadata.copy(),
    )


@dataclass(slots=True, frozen=True)
class CatSessionSnapshot:
  """Immutable snapshot of a session at a specific point in time."""

  session_id: int
  current_theta: float
  true_theta: float | None
  administered_item_ids: list[int]
  responses: list[bool]
  theta_history: list[float]
  status: SessionStatus
  stop_reason: str | None
  metadata: dict[str, Any]


@dataclass(slots=True)
class CatStepResult:
  """Result of applying one CAT step."""

  session: CatSessionSnapshot
  selected_item_id: int | None
  response: bool | None
  updated_theta: float
  stopped: bool
  stop_reason: str | None = None


@dataclass(slots=True)
class ExposureTracker:
  """Track item usage counts during a simulation run."""

  n_items: int
  counts: npt.NDArray[numpy.integer[Any]] = field(init=False)

  def __post_init__(self) -> None:
    """Allocate the item exposure counter array."""
    self.counts = numpy.zeros(self.n_items, dtype=int)

  def record(self, item_id: int) -> None:
    """Record one administration for an item."""
    self.counts[item_id] += 1

  def rates(self, n_sessions: int) -> npt.NDArray[numpy.floating[Any]]:
    """Return exposure rates for the tracked counts."""
    if n_sessions <= 0:
      return numpy.zeros(self.n_items, dtype=float)
    return self.counts.astype(float) / n_sessions


@dataclass(slots=True)
class SimulationResult:
  """Aggregate result for a batch of CAT sessions."""

  item_bank: ItemBank
  sessions: list[CatSessionState]
  exposure_counts: npt.NDArray[numpy.integer[Any]]
  duration: float

  @property
  def exposure_rates(self) -> npt.NDArray[numpy.floating[Any]]:
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
    return [session.latest_theta for session in self.sessions]

  @property
  def examinees(self) -> npt.NDArray[numpy.floating[Any]]:
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
