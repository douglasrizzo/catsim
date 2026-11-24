"""Combine several stoppers into one stopping rule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .base import BaseStopper
from .length import MaxLengthStopper

if TYPE_CHECKING:
  from ..context import StepContext


# __init__ and __repr__ are hand-written because the stoppers are passed as *args, which a generated
# __init__ cannot take and a generated __repr__ would show as a stoppers= keyword
@dataclass(frozen=True, init=False, repr=False)
class StopRule(BaseStopper):
  """Stop as soon as any of several stoppers holds, once the test has a minimum length.

  A rule usually pairs a precision criterion with a :class:`~catsim.stopping.MaxLengthStopper`
  cap, so that every session ends. `min_items` holds back every stopper, the length cap
  included. A rule is itself a stopper, so rules can be nested.

  Parameters
  ----------
  *stoppers : BaseStopper
      The stoppers to combine. At least one is required.
  min_items : int, optional
      Number of administered items before which the rule never stops. Must not be negative,
      nor greater than the `max_items` of a :class:`~catsim.stopping.MaxLengthStopper` among
      `stoppers`, whose cap it would override. Default is 0.

  Raises
  ------
  ValueError
      If no stopper is given, or if `min_items` is negative or greater than the cap of a
      :class:`~catsim.stopping.MaxLengthStopper` among `stoppers`.
  TypeError
      If any of `stoppers` is not a :class:`~catsim.stopping.BaseStopper`.

  Notes
  -----
  For how stoppers combine, see :doc:`/api/techniques/stopping/index`.

  Examples
  --------
  >>> import numpy
  >>> from catsim import ItemBank, StepContext
  >>> from catsim.stopping import MaxLengthStopper, StandardErrorStopper, StopRule
  >>> rule = StopRule(StandardErrorStopper(0.5), MaxLengthStopper(20), min_items=5)
  >>> rule
  StopRule(StandardErrorStopper(max_se=0.5), MaxLengthStopper(max_items=20), min_items=5)
  >>> bank = ItemBank.generate_item_bank(100, seed=0)
  >>> rng = numpy.random.default_rng(0)
  >>> ten = StepContext(item_bank=bank, rng=rng, administered_items=tuple(range(10)), responses=(True,) * 10)
  >>> rule.stop(ten)
  False

  After 20 items the standard error is still above 0.5, so the length cap is what ends the test:

  >>> twenty = StepContext(item_bank=bank, rng=rng, administered_items=tuple(range(20)), responses=(True,) * 20)
  >>> rule.satisfied(twenty)
  (MaxLengthStopper(max_items=20),)
  """

  stoppers: tuple[BaseStopper, ...]
  min_items: int = 0

  def __init__(self, *stoppers: BaseStopper, min_items: int = 0) -> None:
    """Check and store the stoppers and the minimum length."""
    if not stoppers:
      msg = "a StopRule needs at least one stopper"
      raise ValueError(msg)
    for stopper in stoppers:
      if not isinstance(stopper, BaseStopper):
        msg = f"StopRule combines BaseStopper instances, got {type(stopper).__name__}"
        raise TypeError(msg)
    if min_items < 0:
      msg = f"min_items must not be negative, got {min_items}"
      raise ValueError(msg)
    caps = [stopper.max_items for stopper in stoppers if isinstance(stopper, MaxLengthStopper)]
    if caps and min_items > min(caps):
      msg = f"min_items ({min_items}) must not exceed the MaxLengthStopper cap ({min(caps)}), which it would override"
      raise ValueError(msg)
    object.__setattr__(self, "stoppers", stoppers)
    object.__setattr__(self, "min_items", min_items)

  def __repr__(self) -> str:
    """Return the rule as the call that builds it."""
    stoppers = ", ".join(repr(stopper) for stopper in self.stoppers)
    return f"StopRule({stoppers}, min_items={self.min_items})"

  def stop(self, context: StepContext) -> bool:
    """Return whether the test has `min_items` items and any of the stoppers holds.

    Parameters
    ----------
    context : StepContext
        The session to check.

    Returns
    -------
    bool
        True if at least `min_items` items were administered and any stopper holds, False
        otherwise.
    """
    if len(context.administered_items) < self.min_items:
      return False
    return bool(self.satisfied(context))

  def satisfied(self, context: StepContext) -> tuple[BaseStopper, ...]:
    """Return the stoppers that hold for a session, regardless of `min_items`.

    Called with a finished session's items and final estimate, it tells which criteria
    ended the session, for example precision or the length cap.

    Parameters
    ----------
    context : StepContext
        The session to check.

    Returns
    -------
    tuple[BaseStopper, ...]
        The stoppers whose criterion holds, in the order the rule was given them.
    """
    return tuple(stopper for stopper in self.stoppers if stopper.stop(context))
