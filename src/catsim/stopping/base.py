"""Base class for CAT stoppers."""

from abc import ABC, abstractmethod
from typing import Any

from .._base import Simulable


class BaseStopper(Simulable, ABC):
  """Base class for CAT stopping criteria.

  Stoppers determine when a test should end based on specific criteria such as
  test length, measurement precision, or other conditions.
  """

  def __init__(self) -> None:
    """Initialize a Stopper object."""
    super().__init__()

  @abstractmethod
  def stop(self, index: int | None = None, **kwargs: Any) -> bool:
    """Check whether the test reached its stopping criterion for the given user.

    Parameters
    ----------
    index : int or None, optional
        The index of the current examinee in the simulator. When used within a
        simulation, this parameter is provided automatically. When used standalone,
        other parameters may be provided via kwargs. Default is None.
    **kwargs : dict
        Additional keyword arguments that specific Stopper implementations may require.
        Common arguments include:

        - administered_items: Item parameters or indices that were administered
        - theta: Current ability estimate
        - item_bank: ItemBank for accessing item parameters

    Returns
    -------
    bool
        True if the test met its stopping criterion, False otherwise.
    """
