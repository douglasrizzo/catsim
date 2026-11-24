"""Engine primitives for manual and automated CAT execution."""

from __future__ import annotations

import dataclasses
import math
import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy

from .context import Estimate, StepContext
from .exceptions import ExcludedItemError, NoItemsAvailableError
from .item_bank import ItemBank
from .state import CatSessionState, CatStepResult, ExposureTracker, StopReason

if TYPE_CHECKING:
  from collections.abc import Hashable

  from .estimation import BaseEstimator
  from .exposure import BaseExposureControl
  from .initialization import BaseInitializer
  from .selection import BaseSelector
  from .state import CatSessionSnapshot
  from .stopping import BaseStopper


@dataclass(slots=True)
class RunContext:
  """Runtime services shared by CAT sessions in one run.

  ``exposure_tracker`` counts sessions, item administrations and proposals across the run and gives
  selectors the exposure rates their ``r_max`` cap is checked against. Without one, selectors get no
  rates and apply no cap.
  """

  rng: numpy.random.Generator
  exposure_tracker: ExposureTracker | None = None


class ResponseProvider(Protocol):
  """Source of the examinee's responses for :meth:`CatEngine.step` and :meth:`CatEngine.run_session`.

  :class:`~catsim.SimulatedResponseProvider` draws them from a simulated examinee's true
  ability; an application that steps sessions itself can wrap whatever collects real answers.
  """

  def answer(self, item_bank: ItemBank, item_id: int, session: CatSessionSnapshot, rng: numpy.random.Generator) -> bool:
    """Return whether the examinee answers the item correctly.

    Parameters
    ----------
    item_bank : ItemBank
        The engine's item bank.
    item_id : int
        Index of the item the selector chose.
    session : CatSessionSnapshot
        The session as it stands before the response, which the provider cannot change.
    rng : numpy.random.Generator
        The run's random generator, the one the strategies draw from.

    Returns
    -------
    bool
        True for a correct response.
    """


class CatEngine:
  """Engine for stepwise CAT execution on one item bank.

  Every session the engine runs is tested on its item bank: the item indices a session records,
  and the exposure tracker of the run, refer to that bank. An application with several banks
  builds one engine per bank.

  Parameters
  ----------
  item_bank : ItemBank
      The calibrated items the engine's sessions are tested on.
  initializer : BaseInitializer
      Gives each session its initial ability estimate.
  selector : BaseSelector
      Chooses the next item.
  estimator : BaseEstimator
      Updates the ability estimate after each response.
  stopper : BaseStopper
      Decides when a session has ended.
  exposure_control : BaseExposureControl or None, optional
      Decides whether to administer each item the selector proposes; a declined item is excluded
      from the session for the rest of its test. Default is None, which administers every proposed
      item.

  Raises
  ------
  TypeError
      If ``item_bank`` is not an :class:`~catsim.ItemBank`.
  """

  def __init__(
    self,
    item_bank: ItemBank,
    initializer: BaseInitializer,
    selector: BaseSelector,
    estimator: BaseEstimator,
    stopper: BaseStopper,
    *,
    exposure_control: BaseExposureControl | None = None,
  ) -> None:
    # Checked here because a NumPy array of parameters would otherwise fail only at the first session.
    if not isinstance(item_bank, ItemBank):
      msg = f"item_bank must be an ItemBank, got {type(item_bank).__name__}; wrap item parameters in ItemBank(...)"
      raise TypeError(msg)
    self._item_bank = item_bank
    self._initializer = initializer
    self._selector = selector
    self._estimator = estimator
    self._stopper = stopper
    self._exposure_control = exposure_control

  @property
  def item_bank(self) -> ItemBank:
    """Return the item bank the engine's sessions are tested on."""
    return self._item_bank

  def _step_context(self, state: CatSessionState, context: RunContext) -> StepContext:
    """Return what a strategy may read about a session as it stands.

    Parameters
    ----------
    state : CatSessionState
        The session.
    context : RunContext
        Runtime services of the run, which give the random generator and the exposure rates.

    Returns
    -------
    StepContext
        The session's items, responses, excluded items and current estimate with its standard
        error.
    """
    tracker = context.exposure_tracker
    return StepContext(
      item_bank=self._item_bank,
      rng=context.rng,
      administered_items=tuple(state.administered_items),
      responses=tuple(state.responses),
      excluded_items=tuple(state.excluded_items),
      theta=state.current_theta,
      se=state.current_se,
      exposure_rates=None if tracker is None else tracker.rates(),
    )

  def start_session(
    self,
    session_id: Hashable,
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
    context : RunContext
        Runtime services of the run.
    true_theta : float or None, optional
        True ability of a simulated examinee. Default is None.

    Returns
    -------
    CatSessionState
        A running session whose estimate history holds the initial estimate, with an infinite
        standard error.
    """
    initial_theta = float(self._initializer.initialize(item_bank=self._item_bank, rng=context.rng))
    if context.exposure_tracker is not None:
      context.exposure_tracker.start_session()
    initial = Estimate(theta=initial_theta, se=math.inf, evaluations=0)
    return CatSessionState(session_id=session_id, estimates=[initial], true_theta=true_theta)

  def should_stop(self, state: CatSessionState, context: RunContext) -> bool:
    """Return whether the session satisfies the stopping criterion.

    Parameters
    ----------
    state : CatSessionState
        The session to check.
    context : RunContext
        Runtime services of the run.

    Returns
    -------
    bool
        What the stopper says about the session as it stands.
    """
    return bool(self._stopper.stop(self._step_context(state, context)))

  def select_next(self, state: CatSessionState, context: RunContext) -> int | None:
    """Select the next item to administer, or end the session.

    This is the one place that decides whether a session continues, so a manual loop needs no
    other check::

      while (item_id := engine.select_next(state, context)) is not None:
        engine.apply_response(state, item_id, response=ask_examinee(item_id), context=context)

    The stopper is consulted here only for a session that has no responses yet. After that,
    :meth:`apply_response` evaluates it once for every new state, and what it records is final.

    With an exposure control, the items it declines join the session's ``excluded_items`` and the
    exposure tracker's proposals, once the selection has ended, so an application that saves its
    sessions saves them after this call too. A selection that raises leaves both unchanged.

    Each call selects anew, and the exposure control decides anew: a second call before the
    response may decline the item the first returned, or end the session. An application that has
    shown an item keeps its id, which :meth:`apply_response` needs, and shows it again rather than
    calling this method again.

    Parameters
    ----------
    state : CatSessionState
        The session to continue. It is marked stopped when this method returns None for it.
    context : RunContext
        Runtime services of the run: the random generator and, optionally, the exposure tracker.

    Returns
    -------
    int or None
        Index of the next item to administer. None when the session has ended: it was already
        stopped (and is left untouched), its stopper is satisfied (reason ``StopReason.STOPPER``),
        or the selector has no item to offer, or none the exposure control accepts (reason
        ``StopReason.NO_ITEMS``).

    Raises
    ------
    ExcludedItemError
        If the selector proposes an item the session excluded.
    ValueError
        If the selector proposes an item outside the bank or one already administered.
    """
    if state.is_stopped:
      return None
    if state.administered_count == 0 and self.should_stop(state, context):
      state.stop_reason = StopReason.STOPPER
      return None

    declined: list[int] = []
    try:
      item_id: int | None = self._accepted_item(self._step_context(state, context), declined)
    except NoItemsAvailableError:
      item_id = None
    # Committed only here, so a selection that raises changes neither the session nor the tracker.
    state.excluded_items.extend(declined)
    if context.exposure_tracker is not None:
      for declined_item in declined:
        context.exposure_tracker.record_decline(declined_item)
    if item_id is None:
      state.stop_reason = StopReason.NO_ITEMS
    return item_id

  def _accepted_item(self, context: StepContext, declined: list[int]) -> int:
    """Return the first item the selector proposes that the exposure control accepts.

    Parameters
    ----------
    context : StepContext
        The session as it stands.
    declined : list[int]
        Receives each item the control declines, in order.

    Returns
    -------
    int
        The accepted item.

    Raises
    ------
    NoItemsAvailableError
        If the selector has no item left to propose.
    """
    while True:
      # Checked before the control or a response provider sees it: a live provider shows the item to an
      # examinee, and a selector that ignores excluded_items would be asked again forever.
      item_id = self._checked_item(context, int(self._selector.select(context)))
      if self._exposure_control is None or bool(self._exposure_control.accept(context, item_id)):
        return item_id
      declined.append(item_id)
      context = dataclasses.replace(context, excluded_items=(*context.excluded_items, item_id))

  def apply_response(
    self,
    state: CatSessionState,
    item_id: int,
    *,
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
    ExcludedItemError
        If the item is in the session's ``excluded_items``.
    ValueError
        If the session has stopped, the item index is outside the bank or the item was already
        administered in this session.
    TypeError
        If the item index is not an integer, the response is not a ``bool`` or ``numpy.bool_``, or
        the estimator returns something other than an :class:`~catsim.Estimate`.
    """
    if state.is_stopped:
      msg = f"Session {state.session_id} has stopped (reason: {state.stop_reason}) and accepts no more responses"
      raise ValueError(msg)
    # Rejected rather than converted: bool(1.0), bool("no") or bool(None) would silently misscore.
    if not isinstance(response, (bool, numpy.bool_)):
      msg = f"response must be a bool, got {type(response).__name__} {response!r}"
      raise TypeError(msg)
    current = self._step_context(state, context)
    item_id = self._checked_item(current, item_id)

    response = bool(response)
    # The estimator starts from the previous estimate; the stopper judges the new one.
    answered = dataclasses.replace(
      current,
      administered_items=(*current.administered_items, item_id),
      responses=(*current.responses, response),
    )
    estimate = self._estimator.estimate(answered)
    if not isinstance(estimate, Estimate):
      # An estimator written for the float-returning contract would otherwise fail on .theta.
      msg = f"{type(self._estimator).__name__}.estimate must return an Estimate, got {type(estimate).__name__}"
      raise TypeError(msg)
    # The session holds Python scalars, whatever NumPy types a custom estimator puts in its Estimate.
    estimate = Estimate(
      theta=float(estimate.theta),
      se=float(estimate.se),
      evaluations=None if estimate.evaluations is None else int(estimate.evaluations),
    )
    stopped = bool(self._stopper.stop(dataclasses.replace(answered, theta=estimate.theta, se=estimate.se)))

    state.administered_items.append(item_id)
    state.responses.append(response)
    if context.exposure_tracker is not None:
      context.exposure_tracker.record(item_id)
    state.estimates.append(estimate)
    if stopped:
      state.stop_reason = StopReason.STOPPER

    return CatStepResult(session=state.snapshot(), selected_item_id=item_id, response=response)

  def step(
    self,
    state: CatSessionState,
    response_provider: ResponseProvider,
    context: RunContext,
  ) -> CatStepResult:
    """Execute one full CAT step: select an item, get its response and apply it.

    Parameters
    ----------
    state : CatSessionState
        The session to advance.
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
    item_id = self.select_next(state, context)
    if item_id is None:
      return CatStepResult(session=state.snapshot(), selected_item_id=None, response=None)
    response = response_provider.answer(self._item_bank, item_id, state.snapshot(), context.rng)
    return self.apply_response(state, item_id, response=response, context=context)

  def _checked_item(self, context: StepContext, item_id: int) -> int:
    """Return ``item_id`` as an int, raising unless the session in ``context`` may be given it.

    Returns
    -------
    int
        The item index.

    Raises
    ------
    TypeError
        If ``item_id`` is not an integer.
    ExcludedItemError
        If the context excludes the item.
    ValueError
        If ``item_id`` is not an index of the bank, or the session has already had the item.
    """
    try:
      item_id = operator.index(item_id)
    except TypeError:
      msg = f"item_id must be an integer, got {type(item_id).__name__} {item_id!r}"
      raise TypeError(msg) from None
    if item_id < 0 or item_id >= self._item_bank.n_items:
      msg = f"Invalid item index {item_id}. Must be between 0 and {self._item_bank.n_items - 1}"
      raise ValueError(msg)
    if item_id in context.administered_items:
      msg = f"Item {item_id} has already been administered in this session"
      raise ValueError(msg)
    if item_id in context.excluded_items:
      msg = f"Item {item_id} is excluded from this session; remove it from the session's excluded_items to give it"
      raise ExcludedItemError(msg)
    return item_id

  def run_session(
    self,
    state: CatSessionState,
    response_provider: ResponseProvider,
    context: RunContext,
  ) -> CatSessionState:
    """Run a session until a stop condition is reached."""
    while not state.is_stopped:
      self.step(state, response_provider, context)
    return state
