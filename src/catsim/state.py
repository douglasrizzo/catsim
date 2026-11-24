"""Runtime state of CAT sessions and of the runs they belong to."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy

from .context import Estimate

if TYPE_CHECKING:
  from collections.abc import Hashable

  import numpy.typing as npt

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
  int, a str or a UUID. ``estimates`` starts with the initial estimate (which
  :meth:`~catsim.engine.CatEngine.start_session` records with an infinite standard error and no
  evaluations) and gains the :class:`~catsim.context.Estimate` the estimator returned for each
  response; its last entry is the current estimate. A restored session's estimates are trusted
  as given. A session runs
  while ``stop_reason`` is None.

  ``excluded_items`` holds the items the session must not be given although it has not had them:
  those its engine's exposure control declined, in the order it declined them, and any an
  application adds, such as items the examinee saw in an earlier attempt. The engine passes them to
  every strategy and raises :class:`~catsim.exceptions.ExcludedItemError` for a response to one.
  Their indices are checked against the item bank at the session's next step.

  Raises
  ------
  ValueError
      If the session does not have one response per administered item and one more estimate than
      responses, if ``excluded_items`` repeats an item or holds an administered one, or if
      ``stop_reason`` is not a :class:`StopReason` value.
  TypeError
      If an entry of ``estimates`` is not an :class:`~catsim.context.Estimate`.
  """

  session_id: Hashable
  estimates: list[Estimate]
  true_theta: float | None = None
  administered_items: list[int] = field(default_factory=list)
  responses: list[bool] = field(default_factory=list)
  stop_reason: StopReason | None = None
  excluded_items: list[int] = field(default_factory=list)

  def __post_init__(self) -> None:
    """Take its own copy of the histories and reject a session whose entries do not line up."""
    # The engine appends to these lists, so they must be lists the session owns: a snapshot's tuples
    # would fail half-way through a step, and a caller's lists would change under the caller.
    self.administered_items = list(self.administered_items)
    self.responses = list(self.responses)
    self.estimates = list(self.estimates)
    self.excluded_items = list(self.excluded_items)
    if not all(isinstance(estimate, Estimate) for estimate in self.estimates):
      # A session restored from storage needs its estimates rebuilt, not bare thetas or mappings.
      msg = f"Session {self.session_id} needs Estimate entries in estimates, got {self.estimates!r}"
      raise TypeError(msg)
    if self.stop_reason is not None:
      # A stop reason restored from storage comes back as its string value.
      self.stop_reason = StopReason(self.stop_reason)
    n_responses = len(self.responses)
    if len(self.administered_items) != n_responses:
      msg = (
        f"Session {self.session_id} needs one response per administered item, got "
        f"len(administered_items)={len(self.administered_items)} and len(responses)={n_responses}"
      )
      raise ValueError(msg)
    if len(self.estimates) != n_responses + 1:
      msg = (
        f"Session {self.session_id} needs the initial estimate and one estimate per response in "
        f"estimates, got len(estimates)={len(self.estimates)} and len(responses)={n_responses}"
      )
      raise ValueError(msg)
    if len(set(self.excluded_items)) != len(self.excluded_items):
      msg = f"Session {self.session_id} repeats an item in excluded_items, got {self.excluded_items}"
      raise ValueError(msg)
    # A session either gave an item or withheld it: a save that holds an item in both lists is corrupt.
    if administered := sorted(set(self.excluded_items) & set(self.administered_items)):
      msg = f"Session {self.session_id} has administered items {administered}, which excluded_items also holds"
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
    return len(self.administered_items)

  def snapshot(self) -> CatSessionSnapshot:
    """Return an immutable copy of the current session state."""
    return CatSessionSnapshot(
      session_id=self.session_id,
      estimates=tuple(self.estimates),
      true_theta=self.true_theta,
      administered_items=tuple(self.administered_items),
      responses=tuple(self.responses),
      stop_reason=self.stop_reason,
      excluded_items=tuple(self.excluded_items),
    )


@dataclass(slots=True, frozen=True)
class CatSessionSnapshot:
  """Immutable, hashable copy of a session at a specific point in time."""

  session_id: Hashable
  estimates: tuple[Estimate, ...]
  true_theta: float | None
  administered_items: tuple[int, ...]
  responses: tuple[bool, ...]
  stop_reason: StopReason | None
  excluded_items: tuple[int, ...] = ()

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
  def current_theta(self) -> float:
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


@dataclass(slots=True, init=False)
class ExposureTracker:
  """Count item administrations, proposals and sessions across a run, and give sessions their exposure rates.

  One tracker is shared by every session of a run or of a live deployment, through
  :class:`~catsim.engine.RunContext`. The engine calls :meth:`start_session` when a session starts,
  :meth:`record` for every administered item and :meth:`record_decline` for every item the engine's
  exposure control declines. ``counts`` holds each item's administrations so far, those of sessions
  still in progress included, ``proposals`` its administrations plus its declines, and
  ``n_sessions`` the number of sessions started. A session proposes an item at most once, so
  neither count can exceed ``n_sessions``.

  A session is selected against each item's administrations divided by the number of other sessions
  started so far (ADR-0003). With sessions run one after another, as a simulation runs them, that
  is the share of the examinees tested before the current one, as in catR's
  ``simulateRespondents``; the first session sees every rate at 0. Sessions that run side by side,
  as in a live deployment, see each other's administrations as they happen.

  A new tracker starts from zero. An application that must keep the exposure across restarts saves
  ``counts``, ``proposals`` and ``n_sessions`` and passes them back to rebuild the tracker.

  Parameters
  ----------
  n_items : int
      Number of items in the bank.
  counts : array-like of int or None, optional
      Each item's administrations so far, as a saved tracker's ``counts`` holds them. The tracker
      keeps a copy. Default is None, for no administrations.
  n_sessions : int, optional
      Sessions started so far, as a saved tracker's ``n_sessions`` holds them. Default is 0.
  proposals : array-like of int or None, optional
      Each item's proposals so far, as a saved tracker's ``proposals`` holds them. The tracker keeps
      a copy. Default is None, which takes a copy of ``counts``, the proposals of a run without an
      exposure control.

  Raises
  ------
  ValueError
      If ``counts`` or ``proposals`` does not hold one count per item or holds a negative count, if
      an item has more administrations than proposals or more proposals than ``n_sessions``, or if
      ``n_sessions`` is negative.
  TypeError
      If ``counts`` or ``proposals`` holds a value that is not an integer, or ``n_sessions`` is not
      an integer (a bool included).
  """

  n_items: int
  counts: IntArray
  proposals: IntArray
  n_sessions: int

  def __init__(
    self,
    n_items: int,
    counts: npt.ArrayLike | None = None,
    n_sessions: int = 0,
    *,
    proposals: npt.ArrayLike | None = None,
  ) -> None:
    self.n_items = n_items
    self.counts = numpy.zeros(n_items, dtype=int) if counts is None else _restored_counts(n_items, counts, "counts")
    self.proposals = self.counts.copy() if proposals is None else _restored_counts(n_items, proposals, "proposals")
    # bool is an int to operator.index, but a flag passed by mistake is not a session count.
    if isinstance(n_sessions, (bool, numpy.bool_)):
      msg = f"n_sessions must be an integer, got {type(n_sessions).__name__} {n_sessions!r}"
      raise TypeError(msg)
    try:
      self.n_sessions = operator.index(n_sessions)
    except TypeError:
      msg = f"n_sessions must be an integer, got {type(n_sessions).__name__} {n_sessions!r}"
      raise TypeError(msg) from None
    if self.n_sessions < 0:
      msg = f"n_sessions cannot be negative, got {self.n_sessions}"
      raise ValueError(msg)
    # A session administers an item at most once, so no item can have more administrations than sessions.
    if self.counts.size and int(numpy.max(self.counts)) > self.n_sessions:
      msg = (
        f"counts and n_sessions do not belong together: an item has {int(numpy.max(self.counts))} administrations "
        f"but only {self.n_sessions} sessions started"
      )
      raise ValueError(msg)
    # Every administration is a proposal, and an item is proposed at most once per session.
    if (below := numpy.flatnonzero(self.proposals < self.counts)).size:
      item = int(below[0])
      msg = (
        f"counts and proposals do not belong together: item {item} has {int(self.counts[item])} administrations "
        f"but only {int(self.proposals[item])} proposals"
      )
      raise ValueError(msg)
    if self.proposals.size and int(numpy.max(self.proposals)) > self.n_sessions:
      msg = (
        f"proposals and n_sessions do not belong together: an item has {int(numpy.max(self.proposals))} proposals "
        f"but only {self.n_sessions} sessions started"
      )
      raise ValueError(msg)

  def start_session(self) -> None:
    """Count a session that has started."""
    self.n_sessions += 1

  def record(self, item_id: int) -> None:
    """Record one administration for an item, which is also one proposal of it."""
    self.counts[item_id] += 1
    self.proposals[item_id] += 1

  def record_decline(self, item_id: int) -> None:
    """Record one proposal of an item that the exposure control declined; its administrations are unchanged."""
    self.proposals[item_id] += 1

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
        not the run's final exposure, which
        :attr:`~catsim.simulation.SimulationResult.exposure_rates` gives.
    """
    others = self.n_sessions - 1
    if others < 1:
      return numpy.zeros(self.n_items)
    return self.counts / others


def _restored_counts(n_items: int, counts: npt.ArrayLike, name: str) -> IntArray:
  """Return a copy of a saved tracker's per-item counts, checking there is one non-negative integer per item.

  Parameters
  ----------
  n_items : int
      Number of items in the bank.
  counts : array-like of int
      The saved counts.
  name : str
      The tracker attribute they were saved from, for the error messages.

  Returns
  -------
  numpy.ndarray
      A new integer array the tracker owns.

  Raises
  ------
  ValueError
      If ``counts`` does not hold one count per item or holds a negative count.
  TypeError
      If ``counts`` holds a value that is not an integer.
  """
  restored = numpy.array(counts)
  if restored.shape != (n_items,):
    msg = f"{name} must hold one count per item ({n_items}), got shape {restored.shape}"
    raise ValueError(msg)
  # bool is not an integer dtype to NumPy, so a mask passed by mistake is rejected too.
  if not numpy.issubdtype(restored.dtype, numpy.integer):
    msg = f"{name} must be integers, got dtype {restored.dtype}"
    raise TypeError(msg)
  if (restored < 0).any():
    msg = f"{name} cannot be negative, got {restored.min()}"
    raise ValueError(msg)
  # Saved in a compact type such as uint8, the counts would wrap to 0 past its maximum.
  return restored.astype(numpy.int64)
