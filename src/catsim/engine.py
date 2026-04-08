"""Engine primitives for manual and automated CAT execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .exceptions import NoItemsAvailableError
from .irt import icc
from .state import CatSessionState, CatStepResult, ExposureTracker, SessionStatus

if TYPE_CHECKING:
  import numpy

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
    """Select the next item to administer."""
    if state.status == SessionStatus.STOPPED:
      return None
    exposure_rates = None
    if context.exposure_tracker is not None and context.total_sessions is not None:
      exposure_rates = context.exposure_tracker.rates(context.total_sessions)
    return self._selector.select(
      item_bank=item_bank,
      administered_items=state.administered_item_ids,
      est_theta=state.current_theta,
      rng=context.rng,
      exposure_rates=exposure_rates,
    )

  def apply_response(
    self,
    state: CatSessionState,
    item_bank: ItemBank,
    item_id: int,
    response: bool,
    context: RunContext,
  ) -> CatStepResult:
    """Apply a response and update the session state."""
    if item_id < 0 or item_id >= item_bank.n_items:
      msg = f"Invalid item index {item_id}. Must be between 0 and {item_bank.n_items - 1}"
      raise ValueError(msg)
    if item_id in state.administered_item_ids:
      msg = f"Item {item_id} has already been administered in this session"
      raise ValueError(msg)

    state.administered_item_ids.append(item_id)
    state.responses.append(response)

    if context.exposure_tracker is not None:
      context.exposure_tracker.record(item_id)

    updated_theta = self._estimator.estimate(
      item_bank=item_bank,
      administered_items=state.administered_item_ids,
      response_vector=state.responses,
      est_theta=state.current_theta,
    )
    state.current_theta = updated_theta
    state.theta_history.append(updated_theta)

    # After updating the estimate, check whether the completed step satisfied the stop rule.
    stopped = self.should_stop(state, item_bank)
    if stopped:
      state.status = SessionStatus.STOPPED
      state.stop_reason = "stop_rule"

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
    """Execute one full CAT step."""
    # Guard against callers stepping a session that already satisfies the stop rule
    # before item selection begins.
    if self.should_stop(state, item_bank):
      state.status = SessionStatus.STOPPED
      state.stop_reason = "stop_rule"
      return CatStepResult(state.snapshot(), None, None, state.current_theta, True, state.stop_reason)

    try:
      item_id = self.select_next(state, item_bank, context)
    except NoItemsAvailableError:
      state.status = SessionStatus.STOPPED
      state.stop_reason = "item_bank_exhausted"
      return CatStepResult(state.snapshot(), None, None, state.current_theta, True, state.stop_reason)

    if item_id is None:
      state.status = SessionStatus.STOPPED
      state.stop_reason = "selector_stopped"
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
