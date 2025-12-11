"""Base classes for CAT simulation components."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from .simulation import Simulator


class Simulable(ABC):  # noqa: B024
  """Base class representing one of the Simulator components that will receive a reference back to it.

  This class provides the infrastructure for components (Initializers, Selectors,
  Estimators, and Stoppers) to access the Simulator instance they belong to.

  Notes
  -----
  This class inherits from ABC to indicate it's meant to be subclassed, though it doesn't
  define abstract methods itself. Concrete abstract methods are defined in its subclasses
  (Initializer, Selector, Estimator, Stopper).
  """

  def __init__(self) -> None:
    """Initialize a Simulable object."""
    super().__init__()
    self._simulator: Simulator | None = None

  @property
  def simulator(self) -> Simulator:
    """Get the simulator object.

    Returns
    -------
    Simulator
        The :py:class:`Simulator` instance tied to this Simulable.

    Raises
    ------
    TypeError
        If the simulator has not been set.
    """
    if self._simulator is None:
      msg = "simulator has not been set"
      raise TypeError(msg)
    return self._simulator

  @simulator.setter
  def simulator(self, x: Simulator) -> None:
    """Set the simulator object.

    Parameters
    ----------
    x : Simulator
        The Simulator instance to associate with this Simulable.
    """
    self._simulator = x
    self.preprocess()

  def preprocess(self) -> None:  # noqa: B027
    """Override this method to perform any initialization the `Simulable` might need for the simulation.

    `preprocess` is called after a value is set for the `simulator` property. If a new
    value is attributed to `simulator`, this method is called again, guaranteeing that
    internal properties of the `Simulable` are re-initialized as necessary.

    Notes
    -----
    The default implementation does nothing. Subclasses should override this method
    if they need to perform setup operations that require access to the simulator.
    """

  def _prepare_args(
    self,
    return_item_bank: bool = False,
    return_administered_items: bool = False,
    return_response_vector: bool = False,
    return_est_theta: bool = False,
    return_rng: bool = False,
    **kwargs: Any,
  ) -> tuple:
    """Prepare input arguments for all Simulable objects.

    This helper method extracts required arguments either from the simulator (if running
    within a simulation) or from the provided kwargs (if used standalone).

    Parameters
    ----------
    return_item_bank : bool, optional
        Whether to return the ItemBank. Default is False.
    return_administered_items : bool, optional
        Whether to return the list of administered item indices. Default is False.
    return_response_vector : bool, optional
        Whether to return the response vector. Default is False.
    return_est_theta : bool, optional
        Whether to return the estimated theta value. Default is False.
    return_rng : bool, optional
        Whether to return the random number generator. Default is False.
    **kwargs : dict
        Additional keyword arguments that may contain the required values when not
        using a simulator.

    Returns
    -------
    tuple
        A tuple containing the specified results based on the conditions.

    Raises
    ------
    ValueError
        If required arguments are missing when not using a simulator.
    """
    using_simulator_props = kwargs.get("index") is not None and self._simulator is not None
    result = []
    if not using_simulator_props:
      msg = "No simulator in use, but optional arguments missing: "
      missing_args = []

      for ret, val in (
        (return_item_bank, "item_bank"),
        (return_administered_items, "administered_items"),
        (return_response_vector, "response_vector"),
        (return_est_theta, "est_theta"),
        (return_rng, "rng"),
      ):
        if ret:
          try:
            result.append(kwargs[val])
          except KeyError:
            missing_args.append(val)
      if len(missing_args) > 0:
        msg += ", ".join(missing_args)
        raise ValueError(msg)

    else:
      index = kwargs["index"]
      if return_item_bank:
        result.append(self.simulator.item_bank)
      if return_administered_items:
        result.append(self.simulator.administered_items[index])
      if return_response_vector:
        result.append(self.simulator.response_vectors[index])
      if return_est_theta:
        result.append(self.simulator.estimations[index][-1])
      if return_rng:
        result.append(self.simulator.rng)

    return tuple(result)
