"""Base class for exposure controls."""

from abc import ABC, abstractmethod

from ..context import StepContext


class BaseExposureControl(ABC):
  """Base class for exposure controls, which may decline the item a selector proposes.

  A control is an optional part of :class:`~catsim.engine.CatEngine`. The engine asks it about every
  item the selector proposes. A declined item joins the session's ``excluded_items`` for the rest of
  the test, and the engine asks the selector again, until the control accepts an item or the
  selector has none left, which ends the session with ``StopReason.NO_ITEMS``. The run's exposure
  tracker counts each decline as a proposal of the item.

  A control holds its configuration and caches derived from the item bank, never per-session
  state, and draws every random number from ``context.rng``. Here, a control that administers each
  proposed item with probability one half:

  >>> import numpy
  >>> from catsim import CatEngine, ItemBank, RunContext
  >>> from catsim.estimation import EAPEstimator
  >>> from catsim.exposure import BaseExposureControl
  >>> from catsim.initialization import FixedPointInitializer
  >>> from catsim.selection import MaxInfoSelector
  >>> from catsim.stopping import MaxLengthStopper
  >>> class CoinControl(BaseExposureControl):
  ...   def accept(self, context, item_id):
  ...     return bool(context.rng.random() < 0.5)
  >>> bank = ItemBank.generate(20, rng=0)
  >>> engine = CatEngine(
  ...   bank, FixedPointInitializer(0.0), MaxInfoSelector(), EAPEstimator(), MaxLengthStopper(5),
  ...   exposure_control=CoinControl(),
  ... )
  >>> context = RunContext(rng=numpy.random.default_rng(1))
  >>> session = engine.start_session(0, context)
  >>> engine.select_next(session, context), session.excluded_items
  (11, [10, 16])

  The control declined the selector's first two proposals, items 10 and 16, and accepted item 11.
  """

  @abstractmethod
  def accept(self, context: StepContext, item_id: int) -> bool:
    """Return whether the engine administers the item the selector proposed.

    Parameters
    ----------
    context : StepContext
        The context the selector proposed the item from: the session as it stands, with the items
        declined so far in this selection among its ``excluded_items``.
    item_id : int
        Index of the proposed item, an item of ``context.item_bank`` that is neither administered
        nor excluded.

    Returns
    -------
    bool
        True to administer the item; False to decline it for the rest of the test.
    """
