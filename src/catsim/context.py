"""What a per-step strategy may read about the session it serves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import irt

if TYPE_CHECKING:
  import numpy

  from .item_bank import ItemBank
  from .typing import FloatArray


# eq=False: a generated __eq__ would compare the exposure-rate arrays elementwise and raise, and a
# generated __hash__ would fail on them, so contexts compare and hash by identity.
@dataclass(frozen=True, slots=True, eq=False)
class StepContext:
  """What a selector, an estimator or a stopper may read about the session it serves.

  The engine builds a new context for every call: for the selector, from the session as it stands;
  for the estimator, from the session with the newest item and response added and ``theta`` still
  the previous estimate; for the stopper, from that same state with the new estimate. Strategies
  read the fields they need and ignore the rest, so a field added later breaks no strategy. The
  examinee's true ability is deliberately not here: strategies cannot see the simulated truth.

  To call a strategy directly, build a context by hand. Only the item bank and the random
  generator are required; the other fields describe a session with no responses yet:

  >>> import numpy
  >>> from catsim import ItemBank, StepContext
  >>> from catsim.estimation import EAPEstimator
  >>> bank = ItemBank.generate_item_bank(20, seed=0)
  >>> context = StepContext(
  ...   item_bank=bank, rng=numpy.random.default_rng(0), administered_items=(3, 7), responses=(True, False)
  ... )
  >>> round(EAPEstimator().estimate(context), 3)
  -0.427

  Wrappers pass a changed copy on with :func:`dataclasses.replace`, which checks it the same way.

  Parameters
  ----------
  item_bank : ItemBank
      Item bank used by the session.
  rng : numpy.random.Generator
      The run's random generator. Every random draw a strategy makes, tie-breaking included, comes
      from it (ADR-0008).
  administered_items : tuple[int, ...], optional
      Indices of the items administered so far, in order. Their number is the test position.
      Default is no items.
  responses : tuple[bool, ...], optional
      The examinee's responses to ``administered_items``, one per item, in the same order. Default
      is no responses.
  excluded_items : tuple[int, ...], optional
      Items a selector must not choose although they have not been administered, such as items an
      exposure-control wrapper has declined. They do not count toward the test position. The
      engine always passes none.
  theta : float, optional
      The current ability estimate. For the estimator, it is the estimate before the newest
      response, the starting point of its update. Default is 0.
  exposure_rates : numpy.ndarray or None, optional
      Each item's exposure rate over the other sessions of the run (ADR-0003), which selectors
      compare with their ``r_max`` cap. None when the run has no exposure tracker, which means no
      cap applies. Default is None.

  Raises
  ------
  ValueError
      If ``responses`` does not hold one Boolean response per administered item, or an index in
      ``administered_items`` or ``excluded_items`` is outside the item bank.

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
