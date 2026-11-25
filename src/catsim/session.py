"""A CAT session as an immutable value, and why a session ended."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from .checks import _check_disjoint, _item_indices, _responses
from .context import Estimate

if TYPE_CHECKING:
  from collections.abc import Sequence


class StopReason(StrEnum):
  """Why a CAT session ended (ADR-0010).

  ``STOPPER`` means the engine's stopper was satisfied by the session's current state. ``NO_ITEMS``
  means the selector had no item to offer, or none that the exposure control accepted, which is not
  necessarily the end of the item bank: a selector that follows a fixed list or works through
  strata can run out first, and a session whose remaining items are all excluded ends this way too.
  As a ``StrEnum``, each member equals its string value, ``"stopper"`` or ``"no_items"``.
  """

  STOPPER = "stopper"
  NO_ITEMS = "no_items"


@dataclass(frozen=True, slots=True, kw_only=True)
class Session:
  """The record of one examinee's CAT session, as an immutable value (ADR-0006).

  :meth:`CatEngine.start <catsim.CatEngine.start>` returns a new session, and
  :meth:`~catsim.CatEngine.select` and :meth:`~catsim.CatEngine.answer` return the next one; a
  session never changes. It holds plain data only: no true ability, no identifier and no engine,
  which the caller keeps beside it. A session runs while ``stop_reason`` is None.

  An application that stores sessions rebuilds one from the stored fields, which are checked as
  they are when it is built (ADR-0009); the estimates are trusted as given, since only
  recomputation could check them. Item indices are checked against the item bank when the session
  reaches the engine.

  Parameters
  ----------
  administered_items : Sequence[int], optional
      Indices of the items given so far, in order, each given once. Default is no items.
  responses : Sequence[bool], optional
      The responses to the administered items, in the same order: each a ``bool``, a
      ``numpy.bool_`` or the integer 0 or 1, stored as a ``bool``. Default is no responses.
  estimates : Sequence[Estimate]
      The estimate of every state of the session: the initial one, which
      :meth:`~catsim.CatEngine.start` records with an infinite standard error and no evaluations,
      then the one the estimator returned after each response. The last is the current estimate.
  excluded_items : Sequence[int], optional
      Items the session must not be given although it has not had them, in the order they were
      excluded, such as those the examinee saw in an earlier attempt and those the engine's
      exposure control declined (ADR-0011). Default is no items.
  stop_reason : StopReason or None, optional
      Why the session ended, or None while it runs. The reason's string value, as a stored session
      holds it, is accepted. Default is None.

  Raises
  ------
  TypeError
      If an item index is not an integer, a response is neither a Boolean nor an integer, or an
      entry of ``estimates`` is not an :class:`~catsim.Estimate`.
  ValueError
      If an item index is negative or repeats, an item is both administered and excluded, there is
      not one response per administered item and one estimate more than responses, an integer
      response is not 0 or 1, or ``stop_reason`` is not a :class:`StopReason` value.

  Examples
  --------
  A session rebuilt from stored data, with responses stored as 1 and 0:

  >>> import math
  >>> from catsim import Estimate, Session
  >>> session = Session(
  ...   administered_items=[4, 1],
  ...   responses=[1, 0],
  ...   estimates=[Estimate(0.0, math.inf, 0), Estimate(1.2, 1.1, 9), Estimate(0.4, 0.8, 12)],
  ... )
  >>> session.responses, session.current_theta, session.current_se
  ((True, False), 0.4, 0.8)
  >>> Session(administered_items=[4, 4], responses=[1, 0], estimates=session.estimates)
  Traceback (most recent call last):
  ...
  ValueError: administered_items[1] is 4; item indices must not repeat, and administered_items[0] is 4 too
  """

  administered_items: tuple[int, ...] = ()
  responses: tuple[bool, ...] = ()
  estimates: tuple[Estimate, ...]
  excluded_items: tuple[int, ...] = ()
  stop_reason: StopReason | None = None

  def __post_init__(self) -> None:
    """Store the sequences as tuples of catsim's types, after checking that they belong together."""
    administered = _item_indices("administered_items", self.administered_items)
    excluded = _item_indices("excluded_items", self.excluded_items)
    _check_disjoint(administered, excluded)
    responses = _responses(self.responses, len(administered))
    estimates = _estimates(self.estimates, len(responses))
    object.__setattr__(self, "administered_items", administered)
    object.__setattr__(self, "responses", responses)
    object.__setattr__(self, "estimates", estimates)
    object.__setattr__(self, "excluded_items", excluded)
    if self.stop_reason is not None:
      try:
        object.__setattr__(self, "stop_reason", StopReason(self.stop_reason))
      except ValueError:
        reasons = ", ".join(repr(reason.value) for reason in StopReason)
        msg = f"stop_reason is {self.stop_reason!r}; a stop reason must be None or one of {reasons}"
        raise ValueError(msg) from None

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


def _estimates(estimates: Sequence[object], n_responses: int) -> tuple[Estimate, ...]:
  """Return a session's estimates as a tuple, checking that they are the initial one and one per response.

  Parameters
  ----------
  estimates : Sequence[object]
      The estimates, the initial one first.
  n_responses : int
      The number of responses of the session.

  Returns
  -------
  tuple[Estimate, ...]
      The estimates.

  Raises
  ------
  TypeError
      If an entry is not an :class:`~catsim.Estimate`.
  ValueError
      If there is not one estimate more than responses.
  """
  checked: list[Estimate] = []
  for position, estimate in enumerate(estimates):
    # A session restored from storage needs its estimates rebuilt, not bare thetas or mappings.
    if not isinstance(estimate, Estimate):
      msg = f"estimates[{position}] is {estimate!r}; a session's estimates must be Estimate objects"
      raise TypeError(msg)
    checked.append(estimate)
  if len(checked) != n_responses + 1:
    msg = (
      f"estimates has {len(checked)} entries and responses {n_responses}; a session holds the initial estimate and "
      "one estimate per response"
    )
    raise ValueError(msg)
  return tuple(checked)
