"""Engine primitives for manual and automated CAT execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy

from .exceptions import NoItemsAvailableError
from .irt import icc
from .state import CatSessionState, CatStepResult, ExposureTracker, SessionStatus, StopReason

if TYPE_CHECKING:
  from .estimation import BaseEstimator
  from .initialization import BaseInitializer
  from .item_bank import ItemBank
  from .selection import BaseSelector
  from .stopping import BaseStopper


@dataclass(slots=True)
class RunContext:
  """Runtime services shared by CAT sessions in one run."""

  rng: numpy.random.Generator
  exposure_tracker: ExposureTracker | None = None
  total_sessions: int | None = None


class ResponseProvider(Protocol):
  """Protocol for providing responses to administered items."""

  def answer(self, state: CatSessionState, item_bank: ItemBank, item_id: int, context: RunContext) -> bool:
    """Return the response for the selected item."""


class SimulatedResponseProvider:
  """Generate responses from an examinee's true theta."""

  def answer(  # noqa: PLR6301
    self, state: CatSessionState, item_bank: ItemBank, item_id: int, context: RunContext
  ) -> bool:
    """Sample a Bernoulli response using the 4PL model."""
    if state.true_theta is None:
      msg = "true_theta is required to simulate responses"
      raise ValueError(msg)

    item = item_bank.get_item(item_id)
    probability = icc(state.true_theta, item[0], item[1], item[2], item[3])
    return bool(probability >= context.rng.uniform())


class CatEngine:
  """Engine for stepwise CAT execution."""

  def __init__(
    self,
    initializer: BaseInitializer,
    selector: BaseSelector,
    estimator: BaseEstimator,
    stopper: BaseStopper,
  ) -> None:
    self._initializer = initializer
    self._selector = selector
    self._estimator = estimator
    self._stopper = stopper

  def start_session(
    self,
    session_id: int,
    item_bank: ItemBank,
    context: RunContext,
    true_theta: float | None = None,
  ) -> CatSessionState:
    """Create a new CAT session with an initial theta estimate."""
    initial_theta = self._initializer.initialize(item_bank=item_bank, rng=context.rng)
    return CatSessionState(
      session_id=session_id,
      current_theta=initial_theta,
      true_theta=true_theta,
      status=SessionStatus.IN_PROGRESS,
    )

  def should_stop(self, state: CatSessionState, item_bank: ItemBank) -> bool:
    """Return whether the session satisfies the stopping criterion."""
    return self._stopper.stop(
      item_bank=item_bank,
      administered_items=state.administered_item_ids,
      theta=state.current_theta,
    )

  def select_next(self, state: CatSessionState, item_bank: ItemBank, context: RunContext) -> int | None:
    """Select the next item to administer, or end the session.

    This is the one place that decides whether a session continues, so a manual loop needs no
    other check::

      while (item_id := engine.select_next(state, item_bank, context)) is not None:
        engine.apply_response(state, item_bank, item_id, ask_examinee(item_id), context)

    The stopper is consulted here only for a session that has no responses yet. After that,
    :meth:`apply_response` evaluates it once for every new state, and the status it records is
    final.

    Parameters
    ----------
    state : CatSessionState
        The session to continue. It is marked stopped when this method returns None for it.
    item_bank : ItemBank
        Item bank used by the session.
    context : RunContext
        Runtime services of the run: the random generator and, optionally, the exposure tracker.

    Returns
    -------
    int or None
        Index of the next item to administer. None when the session has ended: it was already
        stopped (and is left untouched), its stopper is satisfied (reason ``StopReason.STOPPER``),
        or the selector has no item to offer (reason ``StopReason.NO_ITEMS``).
    """
    if state.status == SessionStatus.STOPPED:
      return None
    if state.administered_count == 0 and self.should_stop(state, item_bank):
      _end(state, StopReason.STOPPER)
      return None

    exposure_rates = None
    if context.exposure_tracker is not None and context.total_sessions is not None:
      exposure_rates = context.exposure_tracker.rates(context.total_sessions)
    try:
      item_id = self._selector.select(
        item_bank=item_bank,
        administered_items=state.administered_item_ids,
        est_theta=state.current_theta,
        rng=context.rng,
        exposure_rates=exposure_rates,
        response_vector=state.responses,
      )
    except NoItemsAvailableError:
      item_id = None
    # BaseSelector.select is typed int | None, so a custom selector may say "no item" with None.
    if item_id is None:
      _end(state, StopReason.NO_ITEMS)
      return None
    return item_id

  def apply_response(
    self,
    state: CatSessionState,
    item_bank: ItemBank,
    item_id: int,
    response: bool,
    context: RunContext,
  ) -> CatStepResult:
    """Record the examinee's response to an item, update the estimate and check the stopper.

    The step is atomic: the inputs are validated and the new estimate is computed before the
    session or the exposure tracker change, so a step that raises leaves both as they were.

    Parameters
    ----------
    state : CatSessionState
        The session that administered the item. It must not be stopped.
    item_bank : ItemBank
        Item bank used by the session.
    item_id : int
        Index of the administered item, usually the one :meth:`select_next` returned.
    response : bool
        Whether the examinee answered the item correctly; a ``bool`` or ``numpy.bool_``.
    context : RunContext
        Runtime services of the run; the item's administration is recorded in its exposure
        tracker, if it has one.

    Returns
    -------
    CatStepResult
        The administered item, the response, the updated estimate and whether the session stopped.

    Raises
    ------
    ValueError
        If the session has stopped, the item index is outside the bank or the item was already
        administered in this session.
    TypeError
        If the response is not a ``bool`` or ``numpy.bool_``.
    """
    if state.status == SessionStatus.STOPPED:
      msg = f"Session {state.session_id} has stopped (reason: {state.stop_reason}) and accepts no more responses"
      raise ValueError(msg)
    # Rejected rather than converted: bool(1.0), bool("no") or bool(None) would silently misscore.
    if not isinstance(response, (bool, numpy.bool_)):
      msg = f"response must be a bool, got {type(response).__name__} {response!r}"
      raise TypeError(msg)
    if item_id < 0 or item_id >= item_bank.n_items:
      msg = f"Invalid item index {item_id}. Must be between 0 and {item_bank.n_items - 1}"
      raise ValueError(msg)
    if item_id in state.administered_item_ids:
      msg = f"Item {item_id} has already been administered in this session"
      raise ValueError(msg)

    response = bool(response)
    updated_theta = self._estimator.estimate(
      item_bank=item_bank,
      administered_items=[*state.administered_item_ids, item_id],
      response_vector=[*state.responses, response],
      est_theta=state.current_theta,
    )

    state.administered_item_ids.append(item_id)
    state.responses.append(response)
    if context.exposure_tracker is not None:
      context.exposure_tracker.record(item_id)
    state.current_theta = updated_theta
    state.theta_history.append(updated_theta)

    stopped = self.should_stop(state, item_bank)
    if stopped:
      _end(state, StopReason.STOPPER)

    return CatStepResult(
      session=state.snapshot(),
      selected_item_id=item_id,
      response=response,
      updated_theta=updated_theta,
      stopped=stopped,
      stop_reason=state.stop_reason,
    )

  def step(
    self,
    state: CatSessionState,
    item_bank: ItemBank,
    response_provider: ResponseProvider,
    context: RunContext,
  ) -> CatStepResult:
    """Execute one full CAT step: select an item, get its response and apply it.

    Parameters
    ----------
    state : CatSessionState
        The session to advance.
    item_bank : ItemBank
        Item bank used by the session.
    response_provider : ResponseProvider
        Source of the examinee's response to the selected item.
    context : RunContext
        Runtime services of the run.

    Returns
    -------
    CatStepResult
        The result of :meth:`apply_response`. When :meth:`select_next` returns None instead, a
        result with no item and no response that reports the session stopped, with the stop
        reason it has (an already stopped session keeps its earlier one).
    """
    item_id = self.select_next(state, item_bank, context)
    if item_id is None:
      return CatStepResult(state.snapshot(), None, None, state.current_theta, True, state.stop_reason)
    response = response_provider.answer(state, item_bank, item_id, context)
    return self.apply_response(state, item_bank, item_id, response, context)

  def run_session(
    self,
    state: CatSessionState,
    item_bank: ItemBank,
    response_provider: ResponseProvider,
    context: RunContext,
  ) -> CatSessionState:
    """Run a session until a stop condition is reached."""
    while state.status != SessionStatus.STOPPED:
      self.step(state, item_bank, response_provider, context)
    return state


def _end(state: CatSessionState, reason: StopReason) -> None:
  state.status = SessionStatus.STOPPED
  state.stop_reason = reason
