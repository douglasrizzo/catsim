"""Base class for exposure controls."""

from abc import ABC, abstractmethod

from ..context import StepContext


class BaseExposureControl(ABC):
  """Base class for exposure controls, which may decline the item a selector proposes.

  :class:`~catsim.CatEngine` asks its control about every item the selector proposes. A
  declined item is excluded from the session for the rest of its test, and the selector proposes
  again; a session with no item left to accept ends with ``StopReason.NO_ITEMS``.
  :meth:`CatEngine.select <catsim.CatEngine.select>` returns the declined items with its selection.
  A control holds configuration only, never per-session state, and draws from ``context.rng``. This
  one administers each proposed item with probability one half:

  >>> from catsim import CatEngine, ItemBank
  >>> from catsim.estimation import EAPEstimator
  >>> from catsim.exposure import BaseExposureControl
  >>> from catsim.initialization import FixedPointInitializer
  >>> from catsim.selection import MaxInfoSelector
  >>> from catsim.stopping import MaxLengthStopper
  >>> class CoinControl(BaseExposureControl):
  ...   def accept(self, context, item):
  ...     return bool(context.rng.random() < 0.5)
  >>> bank = ItemBank.generate(20, rng=0)
  >>> engine = CatEngine(
  ...   bank, FixedPointInitializer(0.0), MaxInfoSelector(), EAPEstimator(), MaxLengthStopper(5),
  ...   exposure_control=CoinControl(),
  ... )
  >>> selection = engine.select(engine.start(rng=1), rng=1)
  >>> selection.item, selection.declined, selection.session.excluded_items
  (11, (10, 16), (10, 16))

  The control declined the selector's first two proposals, items 10 and 16, and accepted item 11.
  """

  @abstractmethod
  def accept(self, context: StepContext, item: int) -> bool:
    """Return whether the engine administers the item the selector proposed.

    Parameters
    ----------
    context : StepContext
        The context the selector proposed the item from: the session as it stands, with the items
        declined so far in this selection among its ``excluded_items``.
    item : int
        Index of the proposed item, an item of ``context.item_bank`` that is neither administered
        nor excluded.

    Returns
    -------
    bool
        True to administer the item; False to decline it for the rest of the test.
    """
