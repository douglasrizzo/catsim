"""Item Bank management for Computerized Adaptive Testing.

This module provides the ItemBank class for managing calibrated item parameters
and precomputing derived values that depend only on static item content.
"""

from enum import StrEnum
from typing import Any, Self, assert_never

import numpy
import numpy.typing as npt
from numpy.random import Generator

from . import irt
from .typing import FloatArray, IntArray

_N_PARAMS = 4
"""Columns of a complete item matrix: a, b, c and d."""


class IrtModel(StrEnum):
  """The logistic model of an item bank, named after the number of parameters its items vary.

  A member equals its name, so ``IrtModel("3PL")`` is ``IrtModel.PL3`` and either can be passed where
  a model is expected. Each model is a special case of the next: a 1PL (Rasch) bank has one
  discrimination for every item, no pseudo-guessing (:math:`c = 0`) and no upper asymptote below 1
  (:math:`d = 1`); a 2PL bank varies the discrimination, a 3PL bank adds pseudo-guessing, and a 4PL
  bank adds upper asymptotes below 1.

  Examples
  --------
  >>> IrtModel("3PL") is IrtModel.PL3
  True
  """

  PL1 = "1PL"
  PL2 = "2PL"
  PL3 = "3PL"
  PL4 = "4PL"


class ItemBank:
  """Container for CAT item parameters with caching of derived values.

  The ItemBank class encapsulates static item parameters, validates them, and
  precomputes expensive calculations that depend only on item parameters
  (like maximum information theta values). This improves performance by
  avoiding redundant calculations during CAT execution.

  Parameters
  ----------
  items : numpy.ndarray
      Item parameter matrix, one row per item, with columns [a, b, c, d]. A matrix with one column
      is read as [b], two as [a, b] and three as [a, b, c], filled with :math:`a = 1`, :math:`c = 0`
      and :math:`d = 1`; a 1-D array is one item.
  validate : bool, optional
      Whether to validate the item parameters. Default is True.

  Attributes
  ----------
  items : numpy.ndarray
      The normalized item parameter matrix (n_items x 4) with columns
      [discrimination, difficulty, pseudo-guessing, upper_asymptote]. It is read-only,
      like the column views returned by the parameter properties. Exposure
      is a property of a run, kept in :py:class:`catsim.simulation.SimulationResult`.
  n_items : int
      Number of items in the bank.
  max_info_thetas : numpy.ndarray
      Cached, read-only theta values where each item provides maximum information.
  max_info_values : numpy.ndarray
      Cached, read-only maximum information values for each item.

  Raises
  ------
  ValueError
      If the matrix does not have 1 to 4 columns, or if validate=True and the matrix has no items or a
      parameter that is NaN, infinite or outside a > 0 and 0 <= c < d <= 1.
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
  >>> bank = ItemBank.generate(100, model="3PL", rng=7)
  >>> bank.n_items
  100
  """

  def __init__(self, items: FloatArray, validate: bool = True) -> None:
    """Initialize an ItemBank with the given item parameters.

    Parameters
    ----------
    items : numpy.ndarray
        Item parameter matrix.
    validate : bool, optional
        Whether to validate the parameters. Default is True.

    Raises
    ------
    TypeError
        If ``items`` is not a numpy.ndarray.
    ValueError
        If ``items`` does not have 1 to 4 columns, or ``validate`` is True and a parameter is invalid.
    """
    # a float copy the caller cannot change afterwards
    self._items = _normalize(items)
    n_columns = self._items.shape[1]
    if n_columns != _N_PARAMS:
      msg = (
        f"Item matrix has {n_columns} columns; ItemBank takes 1 to 4, read as [b], [a, b], [a, b, c] or [a, b, c, d]."
      )
      if n_columns > _N_PARAMS:
        msg += " Earlier versions of catsim kept exposure rates in a fifth column; drop it with items[:, :4]."
      raise ValueError(msg)

    # Validate if requested
    if validate:
      _validate(self._items)
    # Read-only, together with the column views below, so the cached values derived from the
    # parameters cannot go stale
    self._items.flags.writeable = False

    # Lazy-loaded cached values (computed on first access)
    self._max_info_thetas: FloatArray | None = None
    self._max_info_values: FloatArray | None = None
    self._model: IrtModel | None = None

    # Cache for selector-specific data derived from this item bank. Nothing in catsim fills it
    # yet; the planned MPWI and MEPV selectors cache their per-node tables here (see
    # specs/selection/mpwi-selector.md and specs/selection/mepv-selector.md).
    self._selector_cache: dict[str, Any] = {}

  @property
  def items(self) -> FloatArray:
    """Get the item parameter matrix.

    Returns
    -------
    numpy.ndarray
        The read-only item matrix with shape (n_items, 4) containing
        [a, b, c, d] for each item.
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
  def max_info_thetas(self) -> FloatArray:
    """Get the cached theta values where items have maximum information.

    Returns
    -------
    numpy.ndarray
        Read-only array of theta values, one per item.
    """
    if self._max_info_thetas is None:
      thetas = irt.max_info_theta(self._items)
      thetas.flags.writeable = False
      self._max_info_thetas = thetas
    return self._max_info_thetas

  @property
  def max_info_values(self) -> FloatArray:
    """Get the cached maximum information values for each item.

    Returns
    -------
    numpy.ndarray
        Read-only array of maximum information values, one per item.
    """
    if self._max_info_values is None:
      # Information depends on theta only through theta - b, so each item's information at its own
      # peak is its information at 0 once its difficulty is moved to b - peak. Passing the peaks
      # to irt.information would evaluate every peak against every item instead.
      shifted = self._items.copy()
      shifted[:, 1] -= self.max_info_thetas
      values = irt.information(0.0, shifted)
      values.flags.writeable = False
      self._max_info_values = values
    return self._max_info_values

  @property
  def model(self) -> IrtModel:
    """Get the simplest logistic model that fits every item of the bank.

    Returns
    -------
    IrtModel
        ``PL4`` if an upper asymptote is below 1, else ``PL3`` if a pseudo-guessing parameter is above
        0, else ``PL2`` if the discriminations differ, else ``PL1``.
    """
    if self._model is None:
      self._model = _detect_model(self._items)
    return self._model

  @property
  def discrimination(self) -> FloatArray:
    """Get item discrimination parameters (a).

    Returns
    -------
    numpy.ndarray
        Array of discrimination values.
    """
    return self._items[:, 0]

  @property
  def difficulty(self) -> FloatArray:
    """Get item difficulty parameters (b).

    Returns
    -------
    numpy.ndarray
        Array of difficulty values.
    """
    return self._items[:, 1]

  @property
  def pseudo_guessing(self) -> FloatArray:
    """Get item pseudo-guessing parameters (c).

    Returns
    -------
    numpy.ndarray
        Array of pseudo-guessing values.
    """
    return self._items[:, 2]

  @property
  def upper_asymptote(self) -> FloatArray:
    """Get item upper asymptote parameters (d).

    Returns
    -------
    numpy.ndarray
        Array of upper asymptote values.
    """
    return self._items[:, 3]

  def get_item(self, index: int) -> FloatArray:
    """Get parameters for a specific item.

    Parameters
    ----------
    index : int
        The index of the item (0-based).

    Returns
    -------
    numpy.ndarray
        Array containing [a, b, c, d] for the item.

    Raises
    ------
    IndexError
        If index is out of bounds.
    """
    if index < 0 or index >= self.n_items:
      msg = f"Item index {index} out of bounds for bank with {self.n_items} items"
      raise IndexError(msg)
    return self._items[index]

  def get_items(self, indices: npt.ArrayLike) -> FloatArray:
    """Get parameters for multiple items.

    Parameters
    ----------
    indices : npt.ArrayLike
        Array-like (list, tuple, or numpy array) of item indices (int type).

    Returns
    -------
    numpy.ndarray
        Array of item parameters with shape (len(indices), 4).
    """
    return self._items[numpy.asarray(indices, dtype=int)]

  def icc(self, theta: float | FloatArray) -> FloatArray:
    """Return the probability of a correct response to every item, as :py:func:`catsim.irt.icc` gives it.

    Parameters
    ----------
    theta : float or numpy.ndarray
        An ability, or an array of abilities.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_items,)`` for a float ``theta`` and ``(k, n_items)`` for ``k`` abilities.
    """
    return irt.icc(theta, self._items)

  def information(self, theta: float | FloatArray) -> FloatArray:
    """Return the information of every item, as :py:func:`catsim.irt.information` gives it.

    Parameters
    ----------
    theta : float or numpy.ndarray
        An ability, or an array of abilities.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_items,)`` for a float ``theta`` and ``(k, n_items)`` for ``k`` abilities.
    """
    return irt.information(theta, self._items)

  def test_information(self, theta: float, item_indices: npt.ArrayLike | None = None) -> float:
    """Compute test information at a given theta.

    Parameters
    ----------
    theta : float
        Ability value.
    item_indices : npt.ArrayLike or None, optional
        Array-like (list, tuple, or numpy array) of item indices (int type) to include.
        If None, uses all items. Default is None.

    Returns
    -------
    float
        Test information (sum of item information values).
    """
    items_to_use = self._items if item_indices is None else self._items[numpy.asarray(item_indices, dtype=int)]
    return irt.total_information(theta, items_to_use)

  @classmethod
  def generate(
    cls,
    n_items: int,
    model: IrtModel | str = IrtModel.PL4,
    corr: float = 0.0,
    rng: int | Generator = 0,
  ) -> Self:
    """Draw a bank of ``n_items`` items of one logistic model.

    The parameters the model varies are drawn from these distributions; the others are
    :math:`a = 1`, :math:`c = 0` and :math:`d = 1`.

    * discrimination: normal with mean 1.2 and standard deviation 0.25, truncated to positive values;
    * difficulty: standard normal, with correlation ``corr`` to the discrimination;
    * guessing parameter: normal with mean 0.25 and standard deviation 0.02, clipped at 0;
    * upper asymptote: uniform on [0.94, 1].

    The first three are the distributions of the simulated 3PL banks of :cite:t:`Bar10` (p. 445); the
    upper asymptote's is catsim's own. The same seed gives the same bank.

    Parameters
    ----------
    n_items : int
        Number of items.
    model : IrtModel or str, optional
        The logistic model of the items, an :py:class:`IrtModel` or its name in any case ("3pl").
        Default is ``IrtModel.PL4``.
    corr : float, optional
        Correlation between discrimination and difficulty, ignored for ``IrtModel.PL1``. Default is 0.
    rng : int or numpy.random.Generator, optional
        Seed of the random generator the parameters are drawn from, or a generator to draw from,
        which the call advances. Default is 0.

    Returns
    -------
    ItemBank
        The generated bank.

    Raises
    ------
    ValueError
        If ``model`` is not "1PL", "2PL", "3PL" or "4PL", in any case.

    Examples
    --------
    >>> bank = ItemBank.generate(100, model="3PL", corr=0.3, rng=7)
    >>> bank.n_items
    100
    >>> bank.model
    <IrtModel.PL3: '3PL'>
    """
    names = [member.value for member in IrtModel]
    if not isinstance(model, str) or model.upper() not in names:
      msg = f"Invalid model {model!r}; must be one of: {', '.join(names)}"
      raise ValueError(msg)
    model = IrtModel(model.upper())
    generator = numpy.random.default_rng(rng)
    means = [0, 1.2]
    stds = [1, 0.25]
    covs = [
      [stds[0] ** 2, stds[0] * stds[1] * corr],
      [stds[0] * stds[1] * corr, stds[1] ** 2],
    ]

    b, a = generator.multivariate_normal(means, covs, n_items).T

    def redraw_non_positive_discriminations() -> None:
      # redraw the rare non-positive discriminations, together with their difficulties, which samples
      # the bivariate normal truncated to a > 0 and leaves every other item untouched
      invalid = a <= 0
      while invalid.any():
        b[invalid], a[invalid] = generator.multivariate_normal(means, covs, int(invalid.sum())).T
        invalid = a <= 0

    c = numpy.zeros(n_items)
    d = numpy.ones(n_items)
    match model:
      case IrtModel.PL1:
        a = numpy.ones(n_items)
      case IrtModel.PL2:
        redraw_non_positive_discriminations()
      case IrtModel.PL3:
        redraw_non_positive_discriminations()
        c = generator.normal(0.25, 0.02, n_items).clip(min=0)
      case IrtModel.PL4:
        redraw_non_positive_discriminations()
        c = generator.normal(0.25, 0.02, n_items).clip(min=0)
        d = generator.uniform(0.94, 1, n_items)
      case _:
        assert_never(model)

    return cls(numpy.array([a, b, c, d]).T)

  def __repr__(self) -> str:
    """Get string representation of the ItemBank."""
    return f"ItemBank(n_items={self.n_items}, model={self.model})"

  def __len__(self) -> int:
    """Get number of items (allows len(bank))."""
    return self.n_items

  def __setstate__(self, state: dict[str, Any]) -> None:
    """Restore a deep-copied or unpickled bank with its arrays read-only again.

    NumPy does not keep the read-only flag through deepcopy or pickle.
    """
    self.__dict__.update(state)
    for array in (self._items, self._max_info_thetas, self._max_info_values):
      if array is not None:
        array.flags.writeable = False

  def __getitem__(self, index: int | slice | list[int] | IntArray) -> FloatArray:
    """Get item(s) by index (allows bank[i] or bank[indices])."""
    return self._items[index]


def _normalize(items: FloatArray) -> FloatArray:
  r"""Return a float copy of an item matrix, with the parameters a 1PL, 2PL or 3PL matrix leaves out.

  One column is read as [b], two as [a, b] and three as [a, b, c]; the missing discrimination is 1,
  pseudo-guessing 0 and upper asymptote 1. A 1-D array is one item. A matrix with no column or more
  than four is returned as it is, for :py:class:`ItemBank` to reject.

  Parameters
  ----------
  items : numpy.ndarray
      The item matrix.

  Returns
  -------
  numpy.ndarray
      A new float matrix, :math:`n \times 4` unless ``items`` has no column or more than four.

  Raises
  ------
  TypeError
      If ``items`` is not a numpy.ndarray.
  ValueError
      If ``items`` is neither 1-D nor 2-D.
  """
  if not isinstance(items, numpy.ndarray):
    msg = "Item matrix is not of type numpy.ndarray"
    raise TypeError(msg)
  if items.ndim not in {1, 2}:
    msg = f"Item matrix has shape {items.shape}; it must be one item (1-D) or a matrix of items (2-D)"
    raise ValueError(msg)
  matrix = numpy.array(numpy.atleast_2d(items), dtype=float)
  n_columns = matrix.shape[1]
  if not 0 < n_columns < _N_PARAMS:
    return matrix
  complete = numpy.tile([1.0, 0.0, 0.0, 1.0], (matrix.shape[0], 1))
  if n_columns == 1:
    complete[:, 1] = matrix[:, 0]
  else:
    complete[:, :n_columns] = matrix
  return complete


def _validate(items: FloatArray) -> None:
  r"""Check that an item matrix is :math:`n \times 4` with at least one item and valid parameters.

  Every parameter must be finite, with :math:`a > 0` and :math:`0 \leq c < d \leq 1`.

  Parameters
  ----------
  items : numpy.ndarray
      The item matrix to validate.

  Raises
  ------
  ValueError
      If the item matrix is not :math:`n \times 4`, has no items, or has a parameter that is NaN,
      infinite or out of its range. The message lists every parameter violation found, separated by
      ``"; "``.
  """
  if items.shape[1:] != (_N_PARAMS,):
    msg = f"Item matrix has shape {items.shape}; it must be 2-D with exactly 4 columns: a, b, c and d"
    raise ValueError(msg)
  if items.shape[0] == 0:
    msg = "Item matrix has no items"
    raise ValueError(msg)

  errors: list[str] = []
  if not numpy.isfinite(items).all():
    errors.append("There are items with NaN or infinite parameters")
  if any(items[:, 0] <= 0):
    errors.append("There are items with discrimination <= 0")
  if any(items[:, 2] < 0):
    errors.append("There are items with pseudo-guessing < 0")
  if any(items[:, 2] > 1):
    errors.append("There are items with pseudo-guessing > 1")
  if any(items[:, 3] > 1):
    errors.append("There are items with upper asymptote > 1")
  if any(items[:, 3] < 0):
    errors.append("There are items with upper asymptote < 0")
  if any(items[:, 2] >= items[:, 3]):
    errors.append("There are items with pseudo-guessing >= upper asymptote")

  if errors:
    raise ValueError("; ".join(errors))


def _detect_model(items: FloatArray) -> IrtModel:
  """Return the simplest logistic model that fits every item of an n x 4 matrix.

  Parameters
  ----------
  items : numpy.ndarray
      The item matrix.

  Returns
  -------
  IrtModel
      ``PL4`` if an upper asymptote is not 1, else ``PL3`` if a pseudo-guessing parameter is not 0,
      else ``PL2`` if the discriminations differ, else ``PL1``.
  """
  if any(items[:, 3] != 1):
    return IrtModel.PL4
  if any(items[:, 2] != 0):
    return IrtModel.PL3
  if len(set(items[:, 0])) > 1:
    return IrtModel.PL2
  return IrtModel.PL1
