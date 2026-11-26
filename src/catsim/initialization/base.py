"""Base class for CAT initializers."""

from abc import ABC, abstractmethod

import numpy

from ..item_bank import ItemBank


class BaseInitializer(ABC):
  """Base class for CAT initializers.

  An initializer gives a session its initial ability estimate, once, before any item is
  administered. It takes the item bank and the random generator alone, not a
  :class:`~catsim.StepContext`, because no session state exists yet.

  An initializer holds its configuration and caches derived from the item bank, never per-session
  state: one initializer starts every session of a run.
  """

  @abstractmethod
  def initialize(self, *, item_bank: ItemBank, rng: numpy.random.Generator) -> float:
    r"""Return a session's initial :math:`\theta` estimate.

    Parameters
    ----------
    item_bank : ItemBank
        Item bank used by the session.
    rng : numpy.random.Generator
        The generator of the call, for initializers that draw the estimate. When
        :meth:`CatEngine.start <catsim.CatEngine.start>` calls the initializer, it is the one given
        to ``start``.

    Returns
    -------
    float
        The initial :math:`\theta` estimate.
    """
