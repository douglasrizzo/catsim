"""Combine several stoppers into one stopping rule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .base import BaseStopper

if TYPE_CHECKING:
  from ..item_bank import ItemBank


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
      Number of administered items before which the rule never stops. Must not be negative.
      Default is 0.

  Raises
  ------
  ValueError
      If no stopper is given, or if `min_items` is negative.
  TypeError
      If any of `stoppers` is not a :class:`~catsim.stopping.BaseStopper`.

  Notes
  -----
  For how stoppers combine, see :doc:`/api/techniques/stopping/index`.

  Examples
  --------
  >>> from catsim import ItemBank
  >>> from catsim.stopping import MaxLengthStopper, StandardErrorStopper, StopRule
  >>> rule = StopRule(StandardErrorStopper(0.5), MaxLengthStopper(20), min_items=5)
  >>> rule
  StopRule(StandardErrorStopper(max_se=0.5), MaxLengthStopper(max_items=20), min_items=5)
  >>> bank = ItemBank.generate_item_bank(100, seed=0)
  >>> rule.stop(bank, list(range(10)), theta=0.0)
  False

  After 20 items the standard error is still above 0.5, so the length cap is what ends the test:

  >>> rule.satisfied(bank, list(range(20)), theta=0.0)
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
    object.__setattr__(self, "stoppers", stoppers)
    object.__setattr__(self, "min_items", min_items)

  def __repr__(self) -> str:
    """Return the rule as the call that builds it."""
    stoppers = ", ".join(repr(stopper) for stopper in self.stoppers)
    return f"StopRule({stoppers}, min_items={self.min_items})"

  def stop(self, item_bank: ItemBank, administered_items: list[int], theta: float) -> bool:
    """Check whether the test has `min_items` items and any of the stoppers holds.

    Parameters
    ----------
    item_bank : ItemBank
        Item bank used by the session.
    administered_items : list[int]
        Item indices already administered in the session.
    theta : float
        Current ability estimate.

    Returns
    -------
    bool
        True if at least `min_items` items were administered and any stopper holds, False
        otherwise.
    """
    if len(administered_items) < self.min_items:
      return False
    return bool(self.satisfied(item_bank, administered_items, theta))

  def satisfied(self, item_bank: ItemBank, administered_items: list[int], theta: float) -> tuple[BaseStopper, ...]:
    """Return the stoppers that hold for these inputs, regardless of `min_items`.

    Called with a finished session's items and final estimate, it tells which criteria
    ended the session, for example precision or the length cap.

    Parameters
    ----------
    item_bank : ItemBank
        Item bank used by the session.
    administered_items : list[int]
        Item indices already administered in the session.
    theta : float
        Current ability estimate.

    Returns
    -------
    tuple[BaseStopper, ...]
        The stoppers whose criterion holds, in the order the rule was given them.
    """
    return tuple(
      stopper
      for stopper in self.stoppers
      if stopper.stop(item_bank=item_bank, administered_items=administered_items, theta=theta)
    )
