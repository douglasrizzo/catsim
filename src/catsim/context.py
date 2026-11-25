"""What a per-step strategy may read about the session it serves, and what an estimator returns."""

from __future__ import annotations

import dataclasses
import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from . import irt

if TYPE_CHECKING:
  import numpy
  from _typeshed import DataclassInstance

  from .item_bank import ItemBank
  from .typing import FloatArray

_T = TypeVar("_T", bound="DataclassInstance")


# eq=False: a generated __eq__ would compare the exposure-rate arrays elementwise and raise, and a
# generated __hash__ would fail on them, so contexts compare and hash by identity.
@dataclass(frozen=True, slots=True, eq=False)
class StepContext:
  """What a selector, an estimator or a stopper may read about the session it serves.

  The engine builds a new context for every call: for the selector, from the session as it stands;
  for the estimator, with the newest item and response added and ``theta`` still the previous
  estimate; for the stopper, with the new estimate as well. A context holds no true ability, so
  strategies cannot see the simulated truth.

  To call a strategy directly, build a context by hand. Only the item bank and the random
  generator are required; the other fields describe a session with no responses yet:

  >>> import numpy
  >>> from catsim import ItemBank, StepContext
  >>> from catsim.estimation import EAPEstimator
  >>> bank = ItemBank.generate(20, rng=0)
  >>> context = StepContext(
  ...   item_bank=bank, rng=numpy.random.default_rng(0), administered_items=(3, 7), responses=(True, False)
  ... )
  >>> estimate = EAPEstimator().estimate(context)
  >>> round(estimate.theta, 3), round(estimate.se, 3), estimate.evaluations
  (-0.427, 0.978, 121)

  A wrapper passes a changed copy on with :func:`dataclasses.replace`, which checks it the same way.

  Parameters
  ----------
  item_bank : ItemBank
      Item bank used by the session.
  rng : numpy.random.Generator
      The run's random generator. Every random draw a strategy makes, tie-breaking included, comes
      from it.
  administered_items : tuple[int, ...], optional
      Indices of the items administered so far, in order. Their number is the test position.
      Default is no items.
  responses : tuple[bool, ...], optional
      The examinee's responses to ``administered_items``, one per item, in the same order. Default
      is no responses.
  excluded_items : tuple[int, ...], optional
      Items a selector must not choose although they have not been administered: the session's
      excluded items, such as those the engine's exposure control declined. They do not count
      toward the test position. A wrapper that passes a changed context on adds to them, never
      replaces them. Default is no items.
  theta : float, optional
      The current ability estimate. For the estimator, it is the estimate before the newest
      response, the starting point of its update. Default is 0.
  se : float, optional
      The standard error of ``theta``, as its estimator reported it. For the estimator, it is the
      error before the newest response. Default is ``math.inf``, the error before any response.
  exposure_rates : numpy.ndarray or None, optional
      Each item's exposure rate, one value in [0, 1] per item of the bank, which selectors compare
      with their ``r_max`` cap. None means that no cap applies. Default is None.

  Raises
  ------
  ValueError
      If ``responses`` does not hold one response per administered item, or an index in
      ``administered_items`` or ``excluded_items`` is outside the item bank.
  TypeError
      If a response is not a ``bool`` or ``numpy.bool_``.

  Notes
  -----
  Lists given for ``administered_items``, ``responses`` or ``excluded_items`` are stored as
  tuples, so a strategy cannot change the session through its context, and a caller's list cannot
  change under a strategy. Contexts compare equal only to themselves.
  """

  item_bank: ItemBank
  rng: numpy.random.Generator
  administered_items: tuple[int, ...] = ()
  responses: tuple[bool, ...] = ()
  excluded_items: tuple[int, ...] = ()
  theta: float = 0.0
  se: float = math.inf
  exposure_rates: FloatArray | None = None

  def __post_init__(self) -> None:
    """Store the item and response sequences as tuples, and check that they fit each other and the bank."""
    object.__setattr__(self, "administered_items", tuple(self.administered_items))
    object.__setattr__(self, "responses", tuple(self.responses))
    object.__setattr__(self, "excluded_items", tuple(self.excluded_items))
    irt.validate_response_vector(self.responses, len(self.administered_items))
    n_items = self.item_bank.n_items
    for name, items in (("administered_items", self.administered_items), ("excluded_items", self.excluded_items)):
      # NumPy would read a negative index as one counted from the end, and silently drop the wrong item.
      if items and not (min(items) >= 0 and max(items) < n_items):
        msg = f"{name} must hold item indices between 0 and {n_items - 1}, got {items}"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Estimate:
  """An ability estimate, as an estimator returns it and a session records it.

  Parameters
  ----------
  theta : float
      The ability estimate.
  se : float
      Its standard error, by the estimator's own formula: finite, or ``math.inf`` when the
      responses carry no information about the ability, as before any response.
  evaluations : int or None, optional
      How many times the estimator evaluated its objective (the likelihood, the posterior or the
      estimating equation) for this estimate, for comparing the cost of estimators. None where the
      count has no meaning. Default is None.

  Raises
  ------
  ValueError
      If ``theta`` is not finite, ``se`` is negative or NaN, or ``evaluations`` is negative.
  """

  theta: float
  se: float
  evaluations: int | None = None

  def __post_init__(self) -> None:
    """Reject an estimate that the stoppers and the session could not use."""
    # A NaN error would make every comparison False: a confidence-interval stopper would stop at
    # once and a standard-error stopper never.
    if not math.isfinite(self.theta):
      msg = f"theta must be finite, got {self.theta}"
      raise ValueError(msg)
    if not self.se >= 0:
      msg = f"se must be non-negative (math.inf allowed), got {self.se}"
      raise ValueError(msg)
    if self.evaluations is not None and self.evaluations < 0:
      msg = f"evaluations must be non-negative, got {self.evaluations}"
      raise ValueError(msg)


@functools.cache
def _field_defaults(cls: type[DataclassInstance]) -> tuple[tuple[str, object], ...]:
  """Return the name and default of every field of a dataclass, in order."""
  return tuple((field.name, field.default) for field in dataclasses.fields(cls))


def _unchecked(cls: type[_T], /, **fields: object) -> _T:
  """Return an instance of a frozen dataclass built from checked fields, without checking them again.

  The engine builds every session and step context from inputs it has already checked, and the
  simulation runner from inputs it built itself (ADR-0009), so the checks of their ``__post_init__``
  would only repeat work. A field left out takes its default; one without a default raises
  ``KeyError``.

  Parameters
  ----------
  cls : type
      The dataclass, :class:`StepContext` or :class:`~catsim.Session`.
  **fields : object
      The values of its fields, in the types the class stores.

  Returns
  -------
  object
      The instance.
  """
  instance = object.__new__(cls)
  for name, default in _field_defaults(cls):
    value = fields[name] if name in fields or default is dataclasses.MISSING else default
    object.__setattr__(instance, name, value)  # noqa: PLC2801 - the class is frozen
  return instance


def _replaced(instance: _T, /, **changes: object) -> _T:
  """Return a copy of a frozen dataclass with some fields changed, as :func:`dataclasses.replace` does, unchecked.

  Parameters
  ----------
  instance : object
      A :class:`StepContext` or a :class:`~catsim.Session`.
  **changes : object
      The fields to change, in the types the class stores, already checked.

  Returns
  -------
  object
      The copy.
  """
  fields = {name: getattr(instance, name) for name, _ in _field_defaults(type(instance))}
  return _unchecked(type(instance), **(fields | changes))
