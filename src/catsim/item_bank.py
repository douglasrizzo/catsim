"""Item Bank management for Computerized Adaptive Testing.

This module provides the ItemBank class for managing calibrated item parameters and their names,
and precomputing derived values that depend only on static item content.
"""

from collections.abc import Iterable
from enum import StrEnum
from typing import Any, Self, assert_never

import numpy
import numpy.typing as npt
from numpy.random import Generator
from typing_extensions import override

from . import irt
from .typing import FloatArray, IntArray

_N_PARAMS = 4
"""Columns of a complete item matrix: a, b, c and d."""


class IrtModel(StrEnum):
  """The logistic model of an item bank, named after the number of parameters its items vary.

  A member equals its name, so ``IrtModel("3PL")`` is ``IrtModel.PL3`` and either can be passed where
  a model is expected. A 1PL (Rasch) bank has one discrimination for every item, :math:`c = 0` and
  :math:`d = 1`; a 2PL bank varies the discrimination, a 3PL bank also the guessing parameter
  :math:`c`, and a 4PL bank also the upper asymptote :math:`d`.

  Examples
  --------
  >>> IrtModel("3PL") is IrtModel.PL3
  True
  """

  PL1 = "1PL"
  PL2 = "2PL"
  PL3 = "3PL"
  PL4 = "4PL"


class ItemBank:  # noqa: PLR0904 - the core type: parameters, names and derived values
  r"""The calibrated items of a test, with their names and the values derived from their parameters.

  Every item has a name, a non-empty string unique within the bank. Computations refer to an item by
  its row index, from 0 to ``n_items - 1``: ``names[item]`` is the item's name and
  :py:meth:`index` gives a name's row. A bank checks every parameter and name when it is built and
  caches the values that depend only on the parameters, such as each item's point of maximum
  information. Its arrays are read-only, so the cached values cannot go stale. Two banks are equal
  when their parameters and names are. Build a bank from an item matrix, or draw one with
  :py:meth:`generate`.

  Parameters
  ----------
  items : numpy.ndarray
      Item parameter matrix, one row per item, with columns [a, b, c, d]. A matrix with one column
      is read as [b], two as [a, b] and three as [a, b, c], filled with :math:`a = 1`, :math:`c = 0`
      and :math:`d = 1`; a 1-D array is one item.
  names : iterable of str or None, optional
      One name per item, in row order. Default is None, which names each item ``item_`` followed by
      its index, zero-padded to the width of the last index: ``item_000`` to ``item_499`` in a
      500-item bank.

  Attributes
  ----------
  items : numpy.ndarray
      The normalized item parameter matrix (n_items x 4) with columns
      [discrimination, difficulty, guessing, upper_asymptote]. It is read-only,
      like the column views returned by the parameter properties. Exposure
      is a property of a run, kept in :py:class:`catsim.SimulationResult`.
  n_items : int
      Number of items in the bank.
  names : tuple of str
      The items' names, in row order.
  max_info_thetas : numpy.ndarray
      Cached, read-only theta values where each item provides maximum information.
  max_info_values : numpy.ndarray
      Cached, read-only maximum information values for each item.

  Raises
  ------
  ValueError
      If the matrix does not have 1 to 4 columns, has no items, or has a parameter that is NaN,
      infinite or outside :math:`a > 0` and :math:`0 \leq c < d \leq 1`; or if ``names`` does not
      hold one name per item, or holds an empty or repeated name.
  TypeError
      If items is not a numpy.ndarray, or ``names`` is a string or holds a value that is not one.

  Examples
  --------
  >>> import numpy as np
  >>> from catsim import ItemBank
  >>> ItemBank(np.array([[1.0, 0.0, 0.0, 1.0], [1.2, -0.5, 0.1, 1.0]]))
  ItemBank(n_items=2, model=3PL)
  >>> bank = ItemBank(np.array([[0.0], [-0.5], [1.1]]), names=["q7", "q2", "q4"])
  >>> bank.names
  ('q7', 'q2', 'q4')
  >>> bank.index("q2")
  1
  >>> ItemBank.generate(100, model="3PL", rng=7).names[:2]
  ('item_00', 'item_01')
  """

  def __init__(self, items: FloatArray, names: Iterable[str] | None = None) -> None:
    # a float copy the caller cannot change afterwards
    self._items = _normalize(items)
    n_columns = self._items.shape[1]
    if n_columns != _N_PARAMS:
      msg = (
        f"Item matrix has {n_columns} columns; ItemBank takes 1 to 4, read as [b], [a, b], [a, b, c] or [a, b, c, d]."
      )
      if n_columns > _N_PARAMS:
        msg += " Pass items[:, :4] to keep the first four."
      raise ValueError(msg)

    _validate(self._items)
    # Read-only, together with the column views below, so the cached values derived from the
    # parameters cannot go stale
    self._items.flags.writeable = False

    self._index_by_name = _index_names(_generated_names(self.n_items) if names is None else names, self.n_items)
    # dicts keep their insertion order, which is the row order
    self._names = tuple(self._index_by_name)

    # Lazy-loaded cached values (computed on first access)
    self._max_info_thetas: FloatArray | None = None
    self._max_info_values: FloatArray | None = None
    self._model: IrtModel | None = None

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
  def names(self) -> tuple[str, ...]:
    """Get the items' names.

    Returns
    -------
    tuple of str
        One name per item, in row order, so ``names[item]`` is the name of item ``item``.
    """
    return self._names

  def index(self, name: str) -> int:
    """Return the row index of the item with the given name.

    Parameters
    ----------
    name : str
        An item's name.

    Returns
    -------
    int
        The item's index, from 0 to ``n_items - 1``.

    Raises
    ------
    ValueError
        If no item of the bank has that name.
    """
    try:
      return self._index_by_name[name]
    except KeyError:
      msg = f"no item of the bank is named {name!r}"
      raise ValueError(msg) from None

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
        ``PL4`` if an upper asymptote is below 1, else ``PL3`` if a guessing parameter is above 0,
        else ``PL2`` if the discriminations differ, else ``PL1``.
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
    """Get item guessing parameters (c), also called pseudo-guessing parameters.

    Returns
    -------
    numpy.ndarray
        Array of guessing parameter values.
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

  def get_items(self, indices: npt.ArrayLike) -> FloatArray:
    """Return the parameter rows of the given items, in the order of ``indices``.

    Any sequence of item indices is read as a list of items, so a tuple such as a context's
    ``administered_items`` gives its rows and an empty one gives no rows, whereas ``bank[(1, 2)]`` is
    one parameter (row 1, column 2) and ``bank[()]`` is every item.

    Parameters
    ----------
    indices : array-like of int
        Item indices: a list, tuple or integer array, possibly empty.

    Returns
    -------
    numpy.ndarray
        A copy of the rows, shape ``(len(indices), 4)``.

    Examples
    --------
    >>> bank = ItemBank.generate(5, rng=0)
    >>> bank.get_items((1, 2)).shape
    (2, 4)
    >>> bank.get_items(()).shape
    (0, 4)
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

  def total_information(self, theta: float, indices: npt.ArrayLike | None = None) -> float:
    """Return the test information of the given items at ``theta``.

    It is :func:`catsim.irt.total_information` of the items' rows.

    Parameters
    ----------
    theta : float
        Ability value.
    indices : array-like of int or None, optional
        Item indices, as in :meth:`get_items`: a list, tuple or integer array, possibly empty.
        Default is None, every item.

    Returns
    -------
    float
        The sum of the items' information at ``theta``.
    """
    items = self._items if indices is None else self._items[numpy.asarray(indices, dtype=int)]
    return irt.total_information(theta, items)

  @classmethod
  def generate(
    cls,
    n_items: int,
    model: IrtModel | str = IrtModel.PL4,
    corr: float = 0.0,
    rng: int | Generator | None = None,
  ) -> Self:
    """Draw a bank of ``n_items`` items of one logistic model.

    The parameters the model varies are drawn from these distributions; the others are
    :math:`a = 1`, :math:`c = 0` and :math:`d = 1`.

    * discrimination: normal with mean 1.2 and standard deviation 0.25, truncated to positive values;
    * difficulty: standard normal, with correlation ``corr`` to the discrimination;
    * guessing parameter: normal with mean 0.25 and standard deviation 0.02, clipped at 0;
    * upper asymptote: uniform on [0.94, 1].

    The first three are the distributions of the simulated 3PL banks of :cite:t:`Bar10` (p. 445); the
    upper asymptote's is catsim's own.

    Parameters
    ----------
    n_items : int
        Number of items.
    model : IrtModel or str, optional
        The logistic model of the items, an :py:class:`IrtModel` or its name in any case ("3pl").
        Default is ``IrtModel.PL4``.
    corr : float, optional
        Correlation between discrimination and difficulty, ignored for ``IrtModel.PL1``. Default is 0.
    rng : int, numpy.random.Generator or None, optional
        Where the parameters are drawn from. None, the default, draws fresh randomness, so every call
        gives another bank. An int seeds a new generator, so the same seed gives the same bank. A
        generator is drawn from as is, and the call advances it.

    Returns
    -------
    ItemBank
        The generated bank.

    Raises
    ------
    ValueError
        If ``model`` is neither an :py:class:`IrtModel` nor one of the names "1PL", "2PL", "3PL" and
        "4PL", in any case.

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

  @override
  def __repr__(self) -> str:
    """Get string representation of the ItemBank."""
    return f"ItemBank(n_items={self.n_items}, model={self.model})"

  @override
  def __eq__(self, other: object) -> bool:
    """Return whether ``other`` is a bank with the same parameters and names, in the same order."""
    if not isinstance(other, ItemBank):
      return NotImplemented
    return self._names == other._names and bool(numpy.array_equal(self._items, other._items))

  @override
  def __hash__(self) -> int:
    """Return a hash of the parameters and names, equal for equal banks."""
    # -0.0 equals 0.0 but differs in its bytes; adding 0.0 turns it into 0.0
    return hash((self._names, (self._items + 0.0).tobytes()))

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
    """Index the item matrix as NumPy does: ``bank[i]`` is item ``i``'s read-only row, ``bank[-1]`` the last item's."""
    return self._items[index]


def _normalize(items: FloatArray) -> FloatArray:
  r"""Return a float copy of an item matrix, with the parameters a 1PL, 2PL or 3PL matrix leaves out.

  One column is read as [b], two as [a, b] and three as [a, b, c]; the missing discrimination is 1,
  guessing parameter 0 and upper asymptote 1. A 1-D array is one item. A matrix with no column or more
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


def _generated_names(n_items: int) -> list[str]:
  """Return ``item_`` followed by each index, zero-padded to the width of the last index."""
  width = len(str(n_items - 1))
  return [f"item_{item:0{width}d}" for item in range(n_items)]


def _index_names(names: Iterable[str], n_items: int) -> dict[str, int]:
  """Return each item's index by its name, after checking that ``names`` holds one non-empty, unique string per item.

  Parameters
  ----------
  names : iterable of str
      The names, in row order.
  n_items : int
      The number of items.

  Returns
  -------
  dict[str, int]
      Each name's index, in row order.

  Raises
  ------
  TypeError
      If ``names`` is a string, or holds a value that is not one.
  ValueError
      If ``names`` does not hold ``n_items`` names, or holds an empty or repeated one.
  """
  if isinstance(names, str):
    msg = f"names is the string {names!r}; pass one name per item, such as a list of strings"
    raise TypeError(msg)
  given: tuple[object, ...] = tuple(names)
  if len(given) != n_items:
    msg = f"len(names) is {len(given)}, for {n_items} items; names must hold one name per item"
    raise ValueError(msg)
  index_by_name: dict[str, int] = {}
  for item, name in enumerate(given):
    if not isinstance(name, str):
      msg = f"names[{item}] is {name!r}; item names must be strings"
      raise TypeError(msg)
    if not name:
      msg = f"names[{item}] is ''; item names must be non-empty"
      raise ValueError(msg)
    if name in index_by_name:
      msg = f"names[{item}] is {name!r}, as is names[{index_by_name[name]}]; item names must be unique"
      raise ValueError(msg)
    # a plain str, so a name from a NumPy array does not print as np.str_('q1')
    index_by_name[str(name)] = item
  return index_by_name


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

  a, c, d = items[:, 0], items[:, 2], items[:, 3]
  violations = (
    (~numpy.isfinite(items), "There are items with NaN or infinite parameters"),
    (a <= 0, "There are items with discrimination <= 0"),
    (c < 0, "There are items with pseudo-guessing < 0"),
    (c > 1, "There are items with pseudo-guessing > 1"),
    (d > 1, "There are items with upper asymptote > 1"),
    (d < 0, "There are items with upper asymptote < 0"),
    (c >= d, "There are items with pseudo-guessing >= upper asymptote"),
  )
  errors = [message for violated, message in violations if violated.any()]
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
      ``PL4`` if an upper asymptote is not 1, else ``PL3`` if a guessing parameter is not 0,
      else ``PL2`` if the discriminations differ, else ``PL1``.
  """
  if any(items[:, 3] != 1):
    return IrtModel.PL4
  if any(items[:, 2] != 0):
    return IrtModel.PL3
  if len(set(items[:, 0])) > 1:
    return IrtModel.PL2
  return IrtModel.PL1
