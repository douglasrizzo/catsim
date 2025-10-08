"""Item Bank management for Computerized Adaptive Testing.

This module provides the ItemBank class for managing item parameters,
precomputing derived values, and tracking item usage during simulations.
"""

from typing import Any, Literal

import numpy
from numpy.random import Generator
from typing_extensions import Self

from . import irt


class ItemBank:
  """Container for CAT item parameters with caching of derived values.

  The ItemBank class encapsulates item parameters, validates them, and
  precomputes expensive calculations that depend only on item parameters
  (like maximum information theta values). This improves performance by
  avoiding redundant calculations during simulations.

  Parameters
  ----------
  items : numpy.ndarray
      Item parameter matrix with columns [a, b, c, d] or fewer (will be
      normalized to 4 parameters). See :py:func:`catsim.irt.normalize_item_bank`.
  validate : bool, optional
      Whether to validate the item parameters. Default is True.

  Attributes
  ----------
  items : numpy.ndarray
      The normalized item parameter matrix (n_items x 5) with columns
      [discrimination, difficulty, pseudo-guessing, upper_asymptote, exposure_rate].
  n_items : int
      Number of items in the bank.
  max_info_thetas : numpy.ndarray
      Cached theta values where each item provides maximum information.
  max_info_values : numpy.ndarray
      Cached maximum information values for each item.

  Raises
  ------
  ValueError
      If validate=True and item parameters are invalid.
  TypeError
      If items is not a numpy.ndarray.

  Examples
  --------
  >>> import numpy as np
  >>> from catsim.item_bank import ItemBank
  >>> # Create from existing parameters
  >>> params = np.array([[1.0, 0.0, 0.0, 1.0], [1.2, -0.5, 0.1, 1.0]])
  >>> bank = ItemBank(params)
  >>> bank.n_items
  2
  >>> # Generate from distributions
  >>> bank = ItemBank.generate(100, item_type='3PL')
  >>> bank.n_items
  100
  """

  def __init__(self, items: numpy.ndarray, validate: bool = True) -> None:
    """Initialize an ItemBank with the given item parameters.

    Parameters
    ----------
    items : numpy.ndarray
        Item parameter matrix.
    validate : bool, optional
        Whether to validate the parameters. Default is True.
    """
    # Normalize to 4PL format
    self._items = irt.normalize_item_bank(items)

    # Add exposure rate column if not present
    if self._items.shape[1] < 5:  # noqa: PLR2004
      self._items = numpy.append(
        self._items,
        numpy.zeros([self._items.shape[0], 1]),
        axis=1,
      )

    # Validate if requested
    if validate:
      irt.validate_item_bank(self._items[:, :4], raise_err=True)

    # Cache expensive computations
    self._max_info_thetas: numpy.ndarray | None = None
    self._max_info_values: numpy.ndarray | None = None
    self._model: int | None = None

    # Precompute derived values
    self._precompute()

  def _precompute(self) -> None:
    """Precompute and cache values that depend only on item parameters."""
    # Cache maximum information theta values (expensive: sqrt, trig, log operations)
    self._max_info_thetas = irt.max_info_hpc(self._items[:, :4])

    # Cache maximum information values
    self._max_info_values = irt.inf_hpc(self._max_info_thetas, self._items[:, :4])

    # Cache the detected model type
    self._model = irt.detect_model(self._items[:, :4])

  @property
  def items(self) -> numpy.ndarray:
    """Get the item parameter matrix.

    Returns
    -------
    numpy.ndarray
        The item matrix with shape (n_items, 5) containing
        [a, b, c, d, exposure_rate] for each item.
    """
    return self._items

  @property
  def n_items(self) -> int:
    """Get the number of items in the bank.

    Returns
    -------
    int
        Number of items in the bank.
    """
    return self._items.shape[0]

  @property
  def max_info_thetas(self) -> numpy.ndarray:
    """Get the cached theta values where items have maximum information.

    Returns
    -------
    numpy.ndarray
        Array of theta values, one per item.
    """
    return self._max_info_thetas

  @property
  def max_info_values(self) -> numpy.ndarray:
    """Get the cached maximum information values for each item.

    Returns
    -------
    numpy.ndarray
        Array of maximum information values, one per item.
    """
    return self._max_info_values

  @property
  def model(self) -> int:
    """Get the detected IRT model type.

    Returns
    -------
    int
        Integer between 1 and 4 denoting the logistic model:
        1 for 1PL (Rasch), 2 for 2PL, 3 for 3PL, 4 for 4PL.
    """
    return self._model

  @property
  def discrimination(self) -> numpy.ndarray:
    """Get item discrimination parameters (a).

    Returns
    -------
    numpy.ndarray
        Array of discrimination values.
    """
    return self._items[:, 0]

  @property
  def difficulty(self) -> numpy.ndarray:
    """Get item difficulty parameters (b).

    Returns
    -------
    numpy.ndarray
        Array of difficulty values.
    """
    return self._items[:, 1]

  @property
  def pseudo_guessing(self) -> numpy.ndarray:
    """Get item pseudo-guessing parameters (c).

    Returns
    -------
    numpy.ndarray
        Array of pseudo-guessing values.
    """
    return self._items[:, 2]

  @property
  def upper_asymptote(self) -> numpy.ndarray:
    """Get item upper asymptote parameters (d).

    Returns
    -------
    numpy.ndarray
        Array of upper asymptote values.
    """
    return self._items[:, 3]

  @property
  def exposure_rates(self) -> numpy.ndarray:
    """Get item exposure rates.

    Returns
    -------
    numpy.ndarray
        Array of exposure rates (proportion of examinees who received each item).
    """
    return self._items[:, 4]

  def reset_exposure_rates(self) -> None:
    """Reset all item exposure rates to zero.

    This is useful when running multiple simulations with the same item bank.
    """
    self._items[:, 4] = 0

  def reset(self) -> None:
    """Reset the ItemBank to its initial state before any simulations.

    This method:
    - Resets all exposure rates to zero
    - Clears any dynamically cached values computed during simulations
    - Preserves precomputed static values (max_info_thetas, max_info_values)

    This is the recommended method to call before starting a new simulation
    with the same ItemBank instance.

    Examples
    --------
    >>> bank = ItemBank.generate_item_bank(100)
    >>> # ... run simulation ...
    >>> bank.reset()  # Prepare for next simulation
    """
    # Reset exposure rates
    self.reset_exposure_rates()

    # Clear any dynamic caches that selectors might have added
    # (while preserving built-in cached properties like max_info_thetas)
    if hasattr(self, "_selector_cache"):
      self._selector_cache.clear()

  def get_item(self, index: int) -> numpy.ndarray:
    """Get parameters for a specific item.

    Parameters
    ----------
    index : int
        The index of the item (0-based).

    Returns
    -------
    numpy.ndarray
        Array containing [a, b, c, d, exposure_rate] for the item.

    Raises
    ------
    IndexError
        If index is out of bounds.
    """
    if index < 0 or index >= self.n_items:
      msg = f"Item index {index} out of bounds for bank with {self.n_items} items"
      raise IndexError(msg)
    return self._items[index]

  def get_items(self, indices: list[int] | numpy.ndarray) -> numpy.ndarray:
    """Get parameters for multiple items.

    Parameters
    ----------
    indices : list[int] or numpy.ndarray
        Array or list of item indices.

    Returns
    -------
    numpy.ndarray
        Array of item parameters with shape (len(indices), 5).
    """
    return self._items[indices]

  def update_exposure_rate(self, item_index: int, new_rate: float) -> None:
    """Update the exposure rate for a specific item.

    Parameters
    ----------
    item_index : int
        Index of the item to update.
    new_rate : float
        New exposure rate value (should be between 0 and 1).

    Raises
    ------
    IndexError
        If item_index is out of bounds.
    ValueError
        If new_rate is not between 0 and 1.
    """
    if item_index < 0 or item_index >= self.n_items:
      msg = f"Item index {item_index} out of bounds for bank with {self.n_items} items"
      raise IndexError(msg)
    if not 0 <= new_rate <= 1:
      msg = f"Exposure rate must be between 0 and 1, got {new_rate}"
      raise ValueError(msg)
    self._items[item_index, 4] = new_rate

  def icc(self, theta: float | numpy.ndarray) -> numpy.ndarray:
    """Compute item characteristic curves for all items.

    Parameters
    ----------
    theta : float or numpy.ndarray
        Ability value(s).

    Returns
    -------
    numpy.ndarray
        Array of probabilities for correct response. If theta is a scalar,
        returns array of length n_items. If theta is an array, returns
        array of shape (len(theta), n_items).
    """
    if isinstance(theta, (int, float)):
      return irt.icc_hpc(theta, self._items[:, :4])

    # Handle array of thetas
    result = numpy.zeros((len(theta), self.n_items))
    for i, t in enumerate(theta):
      result[i, :] = irt.icc_hpc(t, self._items[:, :4])
    return result

  def information(self, theta: float | numpy.ndarray) -> numpy.ndarray:
    """Compute item information values for all items.

    Parameters
    ----------
    theta : float or numpy.ndarray
        Ability value(s).

    Returns
    -------
    numpy.ndarray
        Array of information values. If theta is a scalar, returns array
        of length n_items. If theta is an array, returns array of shape
        (len(theta), n_items).
    """
    if isinstance(theta, (int, float)):
      return irt.inf_hpc(theta, self._items[:, :4])

    # Handle array of thetas
    result = numpy.zeros((len(theta), self.n_items))
    for i, t in enumerate(theta):
      result[i, :] = irt.inf_hpc(t, self._items[:, :4])
    return result

  def test_information(self, theta: float, item_indices: list[int] | numpy.ndarray | None = None) -> float:
    """Compute test information at a given theta.

    Parameters
    ----------
    theta : float
        Ability value.
    item_indices : list[int] or numpy.ndarray or None, optional
        Indices of items to include. If None, uses all items. Default is None.

    Returns
    -------
    float
        Test information (sum of item information values).
    """
    items_to_use = self._items[:, :4] if item_indices is None else self._items[item_indices, :4]
    return irt.test_info(theta, items_to_use)

  @classmethod
  def generate_item_bank(
    cls,
    n: int,
    itemtype: irt.NumParams | Literal["1PL", "2PL", "3PL", "4PL"] = irt.NumParams.PL4,
    corr: float = 0,
    rng: Generator | None = None,
    seed: int = 0,
    validate: bool = True,
  ) -> Self:
    """Generate a synthetic item bank with parameters following real-world distributions.

    As proposed by [Bar10]_, item parameters are extracted from the following probability
    distributions:

    * discrimination: :math:`N(1.2, 0.25)`
    * difficulty: :math:`N(0,  1)`
    * pseudo-guessing: :math:`N(0.25, 0.02)` (clipped at 0)
    * upper asymptote: :math:`U(0.94, 1)`

    Parameters
    ----------
    n : int
        Number of items to be generated.
    itemtype : irt.NumParams or str, optional
        The logistic model to use: ``irt.NumParams.PL1`` (or "1PL"), ``irt.NumParams.PL2`` (or "2PL"),
        ``irt.NumParams.PL3`` (or "3PL"), or ``irt.NumParams.PL4`` (or "4PL") for the one-, two-, three-,
        or four-parameter logistic models respectively. Default is PL4.
    corr : float, optional
        The correlation between item discrimination and difficulty. If
        ``itemtype == irt.NumParams.PL1``, this parameter is ignored. Default is 0.
    rng : numpy.random.Generator or None, optional
        Optional random number generator to generate the item bank. If not passed,
        one will be created using the seed parameter. Default is None.
    seed : int, optional
        Seed used to create a random number generator if one is not provided.
        Default is 0.
    validate : bool, optional
        Whether to validate item parameters. Default is True.

    Returns
    -------
    ItemBank
        A new ItemBank instance with generated parameters.

    Examples
    --------
    >>> bank = ItemBank.generate_item_bank(100, itemtype='3PL', corr=0.3)
    >>> bank.n_items
    100
    >>> bank.model
    3
    """
    # Convert string to NumParams if needed
    if isinstance(itemtype, str):
      type_map = {
        "1PL": irt.NumParams.PL1,
        "2PL": irt.NumParams.PL2,
        "3PL": irt.NumParams.PL3,
        "4PL": irt.NumParams.PL4,
      }
      itemtype_upper = itemtype.upper()
      if itemtype_upper not in type_map:
        msg = f"Invalid itemtype '{itemtype}'. Must be one of: 1PL, 2PL, 3PL, 4PL"
        raise ValueError(msg)
      itemtype = type_map[itemtype_upper]

    # Generate parameters following standard distributions
    means = [0, 1.2]
    stds = [1, 0.25]
    covs = [
      [stds[0] ** 2, stds[0] * stds[1] * corr],
      [stds[0] * stds[1] * corr, stds[1] ** 2],
    ]

    if rng is None:
      rng = numpy.random.default_rng(seed)

    b, a = rng.multivariate_normal(means, covs, n).T

    if itemtype == irt.NumParams.PL1:
      a = numpy.ones(n)
    elif any(disc < 0 for disc in a):
      # if by chance there is some discrimination value below zero
      # this makes the problem go away
      min_disc = min(a)
      a = [disc + abs(min_disc) for disc in a]

    c = rng.normal(0.25, 0.02, n).clip(min=0) if itemtype in {irt.NumParams.PL3, irt.NumParams.PL4} else numpy.zeros(n)
    d = rng.uniform(0.94, 1, n) if itemtype == irt.NumParams.PL4 else numpy.ones(n)

    # Create ItemBank with normalized parameters
    return cls(irt.normalize_item_bank(numpy.array([a, b, c, d, numpy.zeros(n)]).T), validate=validate)

  @classmethod
  def generate(cls, *args: Any, **kwargs: Any) -> Self:
    """Alias for generate_item_bank for convenience.

    Accepts both 'item_type' and 'itemtype' for backward compatibility.
    """
    # Handle parameter name alias
    if "item_type" in kwargs:
      kwargs["itemtype"] = kwargs.pop("item_type")
    if "n_items" in kwargs:
      kwargs["n"] = kwargs.pop("n_items")
    return cls.generate_item_bank(*args, **kwargs)

  def __repr__(self) -> str:
    """Get string representation of the ItemBank."""
    return f"ItemBank(n_items={self.n_items}, model={self.model}PL)"

  def __len__(self) -> int:
    """Get number of items (allows len(bank))."""
    return self.n_items

  def __getitem__(self, index: int | slice | list | numpy.ndarray) -> numpy.ndarray:
    """Get item(s) by index (allows bank[i] or bank[indices])."""
    return self._items[index]


if __name__ == "__main__":
  import doctest

  doctest.testmod()
