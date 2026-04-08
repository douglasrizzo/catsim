"""Base class for CAT initializers."""

from abc import ABC, abstractmethod
from typing import Any

from ..item_bank import ItemBank


class BaseInitializer(ABC):
  """Base class for CAT initializers.

  Initializers are responsible for selecting examinees' initial ability estimates
  before any items are administered.
  """

  @abstractmethod
  def initialize(self, item_bank: ItemBank, rng: Any, **kwargs: Any) -> float:
    r"""Select an examinee's initial :math:`\theta` value.

    Parameters
    ----------
    **kwargs : dict
        Additional implementation-specific arguments.

    Returns
    -------
    float
        Examinee's initial :math:`\theta` value.
    """
