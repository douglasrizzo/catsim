"""Base class for CAT initializers."""

from abc import ABC, abstractmethod

import numpy

from ..item_bank import ItemBank


class BaseInitializer(ABC):
  """Base class for CAT initializers.

  Initializers are responsible for selecting examinees' initial ability estimates
  before any items are administered.
  """

  @abstractmethod
  def initialize(self, item_bank: ItemBank, rng: numpy.random.Generator) -> float:
    r"""Select an examinee's initial :math:`\theta` value.

    Returns
    -------
    float
        Examinee's initial :math:`\theta` value.
    """
