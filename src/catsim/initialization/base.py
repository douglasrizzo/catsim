"""Base class for CAT initializers."""

from abc import ABC, abstractmethod
from typing import Any

from .._base import Simulable


class BaseInitializer(Simulable, ABC):
  """Base class for CAT initializers.

  Initializers are responsible for selecting examinees' initial ability estimates
  before any items are administered.
  """

  def __init__(self) -> None:
    """Initialize a BaseInitializer object."""
    super().__init__()

  @abstractmethod
  def initialize(self, **kwargs: Any) -> float:
    r"""Select an examinee's initial :math:`\theta` value.

    Parameters
    ----------
    **kwargs : dict
        Arguments used by the BaseInitializer implementation.

    Returns
    -------
    float
        Examinee's initial :math:`\theta` value.
    """
