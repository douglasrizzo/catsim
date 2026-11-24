"""Engine primitives for manual and automated CAT execution."""

from __future__ import annotations

import dataclasses
import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy

from .context import StepContext
from .exceptions import NoItemsAvailableError
from .irt import icc
from .state import CatSessionState, CatStepResult, ExposureTracker, StopReason

if TYPE_CHECKING:
  from collections.abc import Hashable

  from .estimation import BaseEstimator
  from .initialization import BaseInitializer
  from .item_bank import ItemBank
  from .selection import BaseSelector
  from .stopping import BaseStopper


@dataclass(slots=True)
class RunContext:
  """Runtime services shared by CAT sessions in one run.

  ``exposure_tracker`` counts sessions and item administrations across the run and gives selectors
  the exposure rates their ``r_max`` cap is checked against. Without one, selectors get no rates and
  apply no cap.
  """

  rng: numpy.random.Generator
  exposure_tracker: ExposureTracker | None = None


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


def _step_context(state: CatSessionState, item_bank: ItemBank, context: RunContext) -> StepContext:
  """Return what a strategy may read about a session as it stands.

  Parameters
  ----------
  state : CatSessionState
      The session.
  item_bank : ItemBank
      Item bank used by the session.
  context : RunContext
      Runtime services of the run, which give the random generator and the exposure rates.

  Returns
  -------
  StepContext
      The session's items, responses and current estimate, with no excluded items.
  """
  tracker = context.exposure_tracker
  return StepContext(
    item_bank=item_bank,
    rng=context.rng,
    administered_items=tuple(state.administered_item_ids),
    responses=tuple(state.responses),
    theta=state.current_theta,
    exposure_rates=None if tracker is None else tracker.rates(),
  )


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
    session_id: Hashable,
    item_bank: ItemBank,
    context: RunContext,
    true_theta: float | None = None,
  ) -> CatSessionState:
    """Create a new CAT session with an initial theta estimate.

    The session is counted in the context's exposure tracker, if it has one, once the initializer
    has succeeded.

    Parameters
    ----------
    session_id : Hashable
        Identifier of the session, such as an int, a str or a UUID.
    item_bank : ItemBank
        Item bank used by the session.
    context : RunContext
        Runtime services of the run.
    true_theta : float or None, optional
        True ability of a simulated examinee. Default is None.

    Returns
    -------
    CatSessionState
        A running session whose theta history holds the initial estimate.
    """
    initial_theta = float(self._initializer.initialize(item_bank=item_bank, rng=context.rng))
    if context.exposure_tracker is not None:
      context.exposure_tracker.start_session()
    return CatSessionState(session_id=session_id, theta_history=[initial_theta], true_theta=true_theta)

  def should_stop(self, state: CatSessionState, item_bank: ItemBank, context: RunContext) -> bool:
    """Return whether the session satisfies the stopping criterion.

    Parameters
    ----------
    state : CatSessionState
        The session to check.
    item_bank : ItemBank
        Item bank used by the session.
    context : RunContext
        Runtime services of the run.

    Returns
    -------
    bool
        What the stopper says about the session as it stands.
    """
    return bool(self._stopper.stop(_step_context(state, item_bank, context)))

  def select_next(self, state: CatSessionState, item_bank: ItemBank, context: RunContext) -> int | None:
    """Select the next item to administer, or end the session.

    This is the one place that decides whether a session continues, so a manual loop needs no
    other check::

      while (item_id := engine.select_next(state, item_bank, context)) is not None:
        engine.apply_response(state, item_bank, item_id, ask_examinee(item_id), context)

    The stopper is consulted here only for a session that has no responses yet. After that,
    :meth:`apply_response` evaluates it once for every new state, and what it records is final.

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
    if state.is_stopped:
      return None
    if state.administered_count == 0 and self.should_stop(state, item_bank, context):
      state.stop_reason = StopReason.STOPPER
      return None

    try:
      item_id = self._selector.select(_step_context(state, item_bank, context))
    except NoItemsAvailableError:
      state.stop_reason = StopReason.NO_ITEMS
      return None
    return int(item_id)

  def apply_response(
    self,
    state: CatSessionState,
    item_bank: ItemBank,
    item_id: int,
    response: bool,
    context: RunContext,
  ) -> CatStepResult:
    """Record the examinee's response to an item, update the estimate and check the stopper.

    The step is atomic: the inputs are validated, and the new estimate and the stopper's decision
    are computed, before the session or the exposure tracker change, so a step that raises leaves
    both as they were.

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
        If the item index is not an integer, or the response is not a ``bool`` or ``numpy.bool_``.
    """
    if state.is_stopped:
      msg = f"Session {state.session_id} has stopped (reason: {state.stop_reason}) and accepts no more responses"
      raise ValueError(msg)
    # Rejected rather than converted: bool(1.0), bool("no") or bool(None) would silently misscore.
    if not isinstance(response, (bool, numpy.bool_)):
      msg = f"response must be a bool, got {type(response).__name__} {response!r}"
      raise TypeError(msg)
    try:
      item_id = operator.index(item_id)
    except TypeError:
      msg = f"item_id must be an integer, got {type(item_id).__name__} {item_id!r}"
      raise TypeError(msg) from None
    if item_id < 0 or item_id >= item_bank.n_items:
      msg = f"Invalid item index {item_id}. Must be between 0 and {item_bank.n_items - 1}"
      raise ValueError(msg)
    if item_id in state.administered_item_ids:
      msg = f"Item {item_id} has already been administered in this session"
      raise ValueError(msg)

    response = bool(response)
    current = _step_context(state, item_bank, context)
    # The estimator starts from the previous estimate; the stopper judges the new one.
    answered = dataclasses.replace(
      current,
      administered_items=(*current.administered_items, item_id),
      responses=(*current.responses, response),
    )
    updated_theta = float(self._estimator.estimate(answered))
    stopped = bool(self._stopper.stop(dataclasses.replace(answered, theta=updated_theta)))

    state.administered_item_ids.append(item_id)
    state.responses.append(response)
    if context.exposure_tracker is not None:
      context.exposure_tracker.record(item_id)
    state.theta_history.append(updated_theta)
    if stopped:
      state.stop_reason = StopReason.STOPPER

    return CatStepResult(session=state.snapshot(), selected_item_id=item_id, response=response)

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
      return CatStepResult(session=state.snapshot(), selected_item_id=None, response=None)
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
    while not state.is_stopped:
      self.step(state, item_bank, response_provider, context)
    return state
