"""The CAT engine: a test's configuration and the three steps that take a session to the next (ADR-0006)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy
from irtlib import ItemBank

from .checks import _check_in_bank, _exposure_rates, _item, _item_indices, _named, _response
from .context import Estimate, StepContext, _replaced, _unchecked
from .exceptions import ExcludedItemError, NoItemsAvailableError
from .session import Session, StopReason

if TYPE_CHECKING:
  from collections.abc import Iterable

  from irtlib.typing import FloatArray

  from .estimation import BaseEstimator
  from .exposure import BaseExposureControl
  from .initialization import BaseInitializer
  from .selection import BaseSelector
  from .stopping import BaseStopper


@dataclass(frozen=True, slots=True)
class Selection:
  """What :meth:`CatEngine.select` returns: the item to administer, the session and the items declined.

  Parameters
  ----------
  item : int or None
      Index of the item to administer, or None when the session has ended: it had already
      stopped, or the selector had no item to offer, or none the exposure control accepted.
  session : Session
      The session to answer the item in: the one given to :meth:`CatEngine.select`, with the
      declined items added to its excluded items and, when the selector ran out, its
      ``stop_reason`` set to ``StopReason.NO_ITEMS``.
  declined : tuple[int, ...]
      The items the exposure control declined in this selection, in order (ADR-0011). Each is a
      proposal that was not administered, so an application that counts proposals adds them and
      the administered item.
  """

  item: int | None
  session: Session
  declined: tuple[int, ...]


# eq=False: two engines are the same engine only if they are one object, as the strategies they hold
# compare by identity too.
@dataclass(frozen=True, slots=True, eq=False)
class CatEngine:
  """The configuration of a computerized adaptive test, and the steps that run it.

  An engine never changes. Its three step methods take a :class:`~catsim.Session` and return a new
  one, and keep nothing between calls (ADR-0006): :meth:`start` begins a session, :meth:`select`
  proposes its next item and :meth:`answer` records the response to it. The caller keeps the
  sessions, the item on screen and the exposure counts. A session is tested on the engine's item
  bank: its item indices are indices of that bank.

  The stopper is asked once per state of the session: by :meth:`start` about the initial one and by
  :meth:`answer` about each state after a response. A satisfied stopper ends the session with
  ``StopReason.STOPPER``, and a selector with no item to offer ends it in :meth:`select` with
  ``StopReason.NO_ITEMS`` (ADR-0010). The inputs of each step are checked where they enter
  (ADR-0009).

  The :doc:`/embedding-catsim` guide runs a test step by step.

  Parameters
  ----------
  item_bank : ItemBank
      The calibrated items the engine's sessions are tested on.
  initializer : BaseInitializer
      Gives each session its initial ability estimate.
  selector : BaseSelector
      Proposes the next item.
  estimator : BaseEstimator
      Updates the ability estimate after each response.
  stopper : BaseStopper
      Decides when a session has ended.
  exposure_control : BaseExposureControl or None, optional
      Decides whether to administer each item the selector proposes; a declined item is excluded
      from the session for the rest of its test (ADR-0011). Default is None, which administers every
      proposed item.

  Raises
  ------
  TypeError
      If ``item_bank`` is not an ``irtlib.ItemBank``.

  Examples
  --------
  A test of five items, whose examinee answers every item easier than average correctly:

  >>> import numpy
  >>> from catsim import CatEngine
  >>> from irtlib import ItemBank
  >>> from catsim.estimation import EAPEstimator
  >>> from catsim.initialization import FixedPointInitializer
  >>> from catsim.selection import MaxInfoSelector
  >>> from catsim.stopping import MaxLengthStopper
  >>> bank = ItemBank.generate(50, rng=0)
  >>> engine = CatEngine(bank, FixedPointInitializer(0.0), MaxInfoSelector(), EAPEstimator(), MaxLengthStopper(5))
  >>> rng = numpy.random.default_rng(0)
  >>> session = engine.start(rng=rng)
  >>> while True:
  ...   selection = engine.select(session, rng=rng)
  ...   session = selection.session
  ...   if selection.item is None:
  ...     break
  ...   session = engine.answer(session, selection.item, bool(bank.difficulty[selection.item] < 0), rng=rng)
  >>> len(session.administered_items), session.stop_reason
  (5, <StopReason.STOPPER: 'stopper'>)
  """

  item_bank: ItemBank
  initializer: BaseInitializer
  selector: BaseSelector
  estimator: BaseEstimator
  stopper: BaseStopper
  exposure_control: BaseExposureControl | None = field(default=None, kw_only=True)

  def __post_init__(self) -> None:
    """Reject item parameters passed in place of an item bank."""
    # Checked here because a NumPy array of parameters would otherwise fail only at the first session.
    if not isinstance(self.item_bank, ItemBank):
      msg = f"item_bank must be an ItemBank, got {type(self.item_bank).__name__}; wrap item parameters in ItemBank(...)"
      raise TypeError(msg)

  def start(self, *, rng: int | numpy.random.Generator | None, excluded_items: Iterable[int] = ()) -> Session:
    """Begin a session at the initializer's estimate, and ask the stopper about it.

    Parameters
    ----------
    rng : int, numpy.random.Generator or None
        The random source of the initializer and of the stopper's context: a generator, used as
        is, a seed, or None for fresh entropy (ADR-0013).
    excluded_items : Iterable[int], optional
        Items the session must not be given, such as those the examinee saw in an earlier attempt
        or those a recorded examinee has no response to (ADR-0011). Default is no items.

    Returns
    -------
    Session
        A session with no items, whose estimates hold the initial one, with an infinite standard
        error and no evaluations. Its ``stop_reason`` is ``StopReason.STOPPER`` if the stopper is
        satisfied before any item, and None otherwise.

    Raises
    ------
    TypeError
        If an excluded item is not an integer.
    ValueError
        If an excluded item is outside the bank or repeats, or the initializer's estimate is not
        finite.
    """
    excluded = _item_indices("excluded_items", excluded_items, self.item_bank.names)
    return self._start(numpy.random.default_rng(rng), excluded)

  def select(
    self,
    session: Session,
    *,
    rng: int | numpy.random.Generator | None,
    exposure_rates: FloatArray | None = None,
  ) -> Selection:
    """Propose the next item of a session, or end it when the selector has none to offer.

    The selector proposes items until the exposure control accepts one; each item it declines
    joins the session's excluded items (ADR-0011). Every call selects anew, and a selector may draw
    at random, so a second call on the same session may propose another item: keeping the item on
    screen until its response arrives is the caller's.

    Parameters
    ----------
    session : Session
        The session, which ``select`` does not change.
    rng : int, numpy.random.Generator or None
        The random source of the selector and the exposure control: a generator, used as is, a
        seed, or None for fresh entropy (ADR-0013).
    exposure_rates : numpy.ndarray or None, optional
        Each item's exposure rate, one value in [0, 1] per item of the bank, which selectors compare
        with their ``r_max`` cap; the caller computes them from the counts it keeps (ADR-0012).
        Default is None, under which no cap applies.

    Returns
    -------
    Selection
        The item to administer and the session to answer it in, with the items the control
        declined. For a session that has stopped, no item and that same session; for a selector
        with no item to offer, or none the control accepts, no item and the session ended with
        ``StopReason.NO_ITEMS``.

    Raises
    ------
    TypeError
        If ``session`` is not a :class:`~catsim.Session`, the exposure rates are not real numbers or
        the selector proposes an index that is not an integer.
    ValueError
        If an item of the session is outside the bank, the exposure rates are not one value in
        [0, 1] per item, or the selector proposes an item outside the bank or one the session has
        had.
    ExcludedItemError
        If the selector proposes an item the session excluded.
    """
    self._check(session)
    rates = _exposure_rates(exposure_rates, self.item_bank.n_items)
    return self._select(session, numpy.random.default_rng(rng), rates)

  def answer(
    self,
    session: Session,
    item: int,
    response: bool,  # noqa: FBT001 - the response is data, not a flag
    *,
    rng: int | numpy.random.Generator | None,
  ) -> Session:
    """Record the response to an item, update the estimate and ask the stopper about the new state.

    Parameters
    ----------
    session : Session
        The running session the item was administered in, usually ``selection.session``.
    item : int
        Index of the administered item, usually ``selection.item``.
    response : bool
        Whether the examinee answered the item correctly: a ``bool``, a ``numpy.bool_`` or the
        integer 0 or 1.
    rng : int, numpy.random.Generator or None
        The random source of the estimator's and the stopper's contexts: a generator, used as is,
        a seed, or None for fresh entropy (ADR-0013). No built-in estimator or stopper draws from it.

    Returns
    -------
    Session
        The session with the item, the response and the estimator's estimate added. Its
        ``stop_reason`` is ``StopReason.STOPPER`` if the stopper is satisfied by the new state.

    Raises
    ------
    TypeError
        If ``session`` is not a :class:`~catsim.Session`, ``item`` is not an integer, the response
        is neither a Boolean nor an integer, or the estimator returns something other than an
        :class:`~catsim.Estimate`.
    ValueError
        If the session has stopped, an item of the session or ``item`` is outside the bank, the
        session has had ``item``, or an integer response is not 0 or 1.
    ExcludedItemError
        If the session excluded ``item``.
    """
    self._check(session)
    if session.stop_reason is not None:
      msg = f"the session has stopped (stop_reason {session.stop_reason.value!r}) and accepts no more responses"
      raise ValueError(msg)
    checked = self._checked_item("item", item, session)
    return self._answer(session, checked, _response("response", response), numpy.random.default_rng(rng))

  def context(
    self,
    session: Session,
    *,
    rng: int | numpy.random.Generator | None = None,
    exposure_rates: FloatArray | None = None,
  ) -> StepContext:
    """Return what the strategies may read about a session as it stands.

    It is the context the selector reads for the session's next proposal and the stopper read
    about its last state: the session's items, responses, excluded items and current estimate with
    its standard error. A :class:`~catsim.stopping.StopRule` asked with it tells which of its
    criteria hold.

    Parameters
    ----------
    session : Session
        The session.
    rng : int, numpy.random.Generator or None, optional
        The context's random source: a generator, used as is, a seed, or None for fresh entropy
        (ADR-0013). Default is None.
    exposure_rates : numpy.ndarray or None, optional
        The context's exposure rates, one value in [0, 1] per item. Default is None.

    Returns
    -------
    StepContext
        The session's bank, items, responses, excluded items, current estimate and standard error,
        with the generator and the exposure rates.

    Raises
    ------
    TypeError
        If ``session`` is not a :class:`~catsim.Session`, or the exposure rates are not real numbers.
    ValueError
        If an item of the session is outside the bank, or the exposure rates are not one value in
        [0, 1] per item.

    Examples
    --------
    >>> from catsim import CatEngine, simulate
    >>> from irtlib import ItemBank
    >>> from catsim.estimation import EAPEstimator
    >>> from catsim.initialization import FixedPointInitializer
    >>> from catsim.selection import MaxInfoSelector
    >>> from catsim.stopping import MaxLengthStopper, StandardErrorStopper, StopRule
    >>> rule = StopRule(StandardErrorStopper(0.2), MaxLengthStopper(5))
    >>> bank = ItemBank.generate(50, rng=0)
    >>> engine = CatEngine(bank, FixedPointInitializer(0.0), MaxInfoSelector(), EAPEstimator(), rule)
    >>> session = simulate(engine, true_thetas=[0.0], rng=0).sessions[0]
    >>> rule.satisfied(engine.context(session))
    (MaxLengthStopper(max_items=5),)
    """
    self._check(session)
    rates = _exposure_rates(exposure_rates, self.item_bank.n_items)
    return self._context(session, numpy.random.default_rng(rng), rates)

  def _check(self, session: Session) -> None:
    """Raise unless ``session`` is a session whose items are items of the engine's bank.

    Raises
    ------
    TypeError
        If ``session`` is not a :class:`~catsim.Session`.
    """
    if not isinstance(session, Session):
      msg = f"session is a {type(session).__name__}; it must be a Session"
      raise TypeError(msg)
    # The session checked everything else when it was built, but it does not know the bank.
    _check_in_bank("administered_items", session.administered_items, self.item_bank.n_items)
    _check_in_bank("excluded_items", session.excluded_items, self.item_bank.n_items)

  def _checked_item(self, label: str, value: object, session: Session | StepContext) -> int:
    """Return ``value`` as an item index, raising unless the session may be given that item.

    Parameters
    ----------
    label : str
        What the value is, for the messages.
    value : object
        The item index.
    session : Session or StepContext
        The session, or the context it is proposed in, whose items and exclusions it is checked
        against.

    Returns
    -------
    int
        The item index.

    Raises
    ------
    ExcludedItemError
        If the session excluded the item.
    ValueError
        If the session has had the item.
    """
    item = _item(label, value, self.item_bank.n_items)
    if item in session.administered_items:
      msg = f"{label} is {_named(item, self.item_bank.names)}; the session has already been given that item"
      raise ValueError(msg)
    if item in session.excluded_items:
      msg = (
        f"{label} is {_named(item, self.item_bank.names)}; the session excluded that item, which it must not be given"
      )
      raise ExcludedItemError(msg)
    return item

  # The steps below take checked inputs and check nothing again, so simulate calls
  # them directly with the inputs it builds itself (ADR-0009).

  def _start(self, rng: numpy.random.Generator, excluded_items: tuple[int, ...]) -> Session:
    """Return a new session excluding ``excluded_items``, stopped if the stopper holds before any item."""
    theta = float(self.initializer.initialize(item_bank=self.item_bank, rng=rng))
    initial = Estimate(theta=theta, se=math.inf, evaluations=0)
    session = _unchecked(Session, estimates=(initial,), excluded_items=excluded_items)
    if bool(self.stopper.stop(self._context(session, rng, None))):
      return _replaced(session, stop_reason=StopReason.STOPPER)
    return session

  def _select(self, session: Session, rng: numpy.random.Generator, exposure_rates: FloatArray | None) -> Selection:
    """Return the selection for ``session``, ending it with ``NO_ITEMS`` when no item is accepted."""
    if session.stop_reason is not None:
      return Selection(item=None, session=session, declined=())
    item, declined = self._accepted(self._context(session, rng, exposure_rates))
    if item is None:
      excluded = (*session.excluded_items, *declined)
      session = _replaced(session, excluded_items=excluded, stop_reason=StopReason.NO_ITEMS)
    elif declined:
      session = _replaced(session, excluded_items=(*session.excluded_items, *declined))
    return Selection(item=item, session=session, declined=declined)

  def _accepted(self, context: StepContext) -> tuple[int | None, tuple[int, ...]]:
    """Return the first proposal the exposure control accepts, or None, with the proposals it declined before.

    Returns
    -------
    tuple[int or None, tuple[int, ...]]
        The accepted item, or None if the selector ran out first, and the declined items, in order.
    """
    declined: list[int] = []
    while True:
      try:
        # Checked before the control sees it: a selector that ignores excluded_items would be asked
        # about a declined item forever.
        item = self._checked_item(
          f"the item {type(self.selector).__name__} proposed", self.selector.select(context), context
        )
        accepted = self.exposure_control is None or bool(self.exposure_control.accept(context, item))
      except NoItemsAvailableError:
        return None, tuple(declined)
      if accepted:
        return item, tuple(declined)
      declined.append(item)
      context = _replaced(context, excluded_items=(*context.excluded_items, item))

  def _answer(self, session: Session, item: int, response: bool, rng: numpy.random.Generator) -> Session:  # noqa: FBT001
    """Return ``session`` with the response to ``item`` and the new estimate, stopped if the stopper holds."""
    # The estimator starts from the previous estimate; the stopper judges the new one.
    answered = _replaced(
      self._context(session, rng, None),
      administered_items=(*session.administered_items, item),
      responses=(*session.responses, response),
    )
    estimate = self._estimate(answered)
    stopped = bool(self.stopper.stop(_replaced(answered, theta=estimate.theta, se=estimate.se)))
    return _unchecked(
      Session,
      administered_items=answered.administered_items,
      responses=answered.responses,
      estimates=(*session.estimates, estimate),
      excluded_items=session.excluded_items,
      stop_reason=StopReason.STOPPER if stopped else None,
    )

  def _estimate(self, context: StepContext) -> Estimate:
    """Return the estimator's estimate for ``context``, holding Python scalars.

    Returns
    -------
    Estimate
        The estimate, its theta and standard error as floats and its evaluations as an int or None.

    Raises
    ------
    TypeError
        If the estimator returns something other than an :class:`~catsim.Estimate`.
    """
    estimate = self.estimator.estimate(context)
    if not isinstance(estimate, Estimate):
      # An estimator written for the float-returning contract would otherwise fail on .theta.
      msg = f"{type(self.estimator).__name__}.estimate must return an Estimate, got {type(estimate).__name__}"
      raise TypeError(msg)
    # The session holds Python scalars, whatever NumPy types a custom estimator puts in its Estimate.
    return Estimate(
      theta=float(estimate.theta),
      se=float(estimate.se),
      evaluations=None if estimate.evaluations is None else int(estimate.evaluations),
    )

  def _context(self, session: Session, rng: numpy.random.Generator, exposure_rates: FloatArray | None) -> StepContext:
    """Return the context of ``session`` as it stands, built from its checked fields without checking them again."""
    return _unchecked(
      StepContext,
      item_bank=self.item_bank,
      rng=rng,
      administered_items=session.administered_items,
      responses=session.responses,
      excluded_items=session.excluded_items,
      theta=session.current_theta,
      se=session.current_se,
      exposure_rates=exposure_rates,
    )
