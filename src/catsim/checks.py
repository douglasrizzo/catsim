"""The checks of the explicit inputs of a step, made where they enter catsim (ADR-0009).

A failed check raises ``TypeError`` or ``ValueError`` with a message that names the input, the
position, the value and the rule, such as "administered_items[1] is 3.7; item indices must be
integers". Nothing is converted silently: an accepted value is only stored in catsim's own types,
such as a ``numpy.int64`` index as an ``int`` and a response of 1 as ``True``.

The functions are internal: :class:`~catsim.Session`, :class:`~catsim.StepContext`,
:class:`~catsim.CatEngine` and the functions of :mod:`catsim.irt` that take responses call them
on the inputs they receive.
"""

from __future__ import annotations

import math
import numbers
import operator
from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
  from collections.abc import Iterable

  from .typing import FloatArray


def _item(label: str, value: object, n_items: int | None) -> int:
  """Return one item index as an int, checking that it is an integer of the bank.

  Parameters
  ----------
  label : str
      What the index is, for the message: an input and a position, such as
      ``"administered_items[2]"``, or an argument, such as ``"item"``.
  value : object
      The index.
  n_items : int or None
      The number of items in the bank, or None where the bank is not known, which checks only that
      the index is not negative.

  Returns
  -------
  int
      The index.

  Raises
  ------
  TypeError
      If the index is not an integer, a bool included.
  ValueError
      If it is negative, or not below ``n_items``.
  """
  # bool is an int to operator.index, but a flag stored by mistake is not an item.
  if isinstance(value, (bool, numpy.bool_)):
    msg = f"{label} is {value!r}; item indices must be integers, not bools"
    raise TypeError(msg)
  try:
    item = operator.index(value)  # ty: ignore[invalid-argument-type]
  except TypeError:
    # 3.7 would read as item 3 where NumPy indexes the bank with it.
    msg = f"{label} is {value!r}; item indices must be integers"
    raise TypeError(msg) from None
  # NumPy would read a negative index as one counted from the end of the bank.
  if item < 0 or (n_items is not None and item >= n_items):
    bound = "must not be negative" if n_items is None else f"must be between 0 and {n_items - 1}"
    msg = f"{label} is {item}; item indices {bound}"
    raise ValueError(msg)
  return item


def _item_indices(name: str, items: Iterable[object], n_items: int | None = None) -> tuple[int, ...]:
  """Return item indices as a tuple of ints, checking that they are distinct integers of the bank.

  Parameters
  ----------
  name : str
      The input the indices are, such as ``"administered_items"``, for the messages.
  items : Iterable[object]
      The indices, in order.
  n_items : int or None, optional
      The number of items in the bank, or None where the bank is not known, which checks only that
      no index is negative. Default is None.

  Returns
  -------
  tuple[int, ...]
      The indices, in order.

  Raises
  ------
  TypeError
      If an index is not an integer.
  ValueError
      If an index is outside the bank, or repeats an earlier one.
  """
  indices = tuple(_item(f"{name}[{position}]", value, n_items) for position, value in enumerate(items))
  if len(set(indices)) != len(indices):
    # A repeated item counts twice in every estimate after it.
    first: dict[int, int] = {}
    for position, item in enumerate(indices):
      if item in first:
        msg = f"{name}[{position}] is {item}; item indices must not repeat, and {name}[{first[item]}] is {item} too"
        raise ValueError(msg)
      first[item] = position
  return indices


def _check_in_bank(name: str, items: tuple[int, ...], n_items: int) -> None:
  """Check that checked item indices, which may come from a session built without the bank, are in the bank.

  Parameters
  ----------
  name : str
      The input the indices belong to, for the message.
  items : tuple[int, ...]
      Non-negative item indices.
  n_items : int
      The number of items in the bank.

  Raises
  ------
  ValueError
      If an index is not below ``n_items``.
  """
  if items and max(items) >= n_items:
    position = next(position for position, item in enumerate(items) if item >= n_items)
    msg = f"{name}[{position}] is {items[position]}; item indices must be between 0 and {n_items - 1}"
    raise ValueError(msg)


def _check_disjoint(administered_items: tuple[int, ...], excluded_items: tuple[int, ...]) -> None:
  """Check that no item is both administered and excluded.

  Parameters
  ----------
  administered_items : tuple[int, ...]
      The items given, in order.
  excluded_items : tuple[int, ...]
      The items withheld, in order.

  Raises
  ------
  ValueError
      If an excluded item was also administered.
  """
  if not excluded_items or set(administered_items).isdisjoint(excluded_items):
    return
  position, item = next((position, item) for position, item in enumerate(excluded_items) if item in administered_items)
  msg = (
    f"excluded_items[{position}] is {item}; an item cannot be both administered and excluded, and "
    f"administered_items[{administered_items.index(item)}] is {item} too"
  )
  raise ValueError(msg)


def _response(name: str, value: object) -> bool:
  """Return one response as a bool, checking that it is a Boolean or the integer 0 or 1.

  Parameters
  ----------
  name : str
      The input the response is, with its position, such as ``"responses[2]"``, for the message.
  value : object
      The response: True or 1 for a correct answer, False or 0 for an incorrect one.

  Returns
  -------
  bool
      The response.

  Raises
  ------
  TypeError
      If the response is neither a ``bool``, a ``numpy.bool_`` nor an integer.
  ValueError
      If it is an integer other than 0 and 1.
  """
  if isinstance(value, (bool, numpy.bool_)):
    return bool(value)
  # Rejected rather than converted: bool(0.5), bool("no") and bool(None) would silently misscore.
  if not isinstance(value, (int, numpy.integer)):
    msg = f"{name} is {value!r}; responses must be bools or the integers 0 and 1"
    raise TypeError(msg)
  if value not in {0, 1}:
    msg = f"{name} is {value!r}; an integer response must be 0 or 1"
    raise ValueError(msg)
  return bool(value)


def _responses(values: Iterable[object], n_items: int) -> tuple[bool, ...]:
  """Return the responses as a tuple of bools, checking that there is one valid response per administered item.

  Parameters
  ----------
  values : Iterable[object]
      The responses, in the order of the administered items.
  n_items : int
      The number of administered items.

  Returns
  -------
  tuple[bool, ...]
      The responses.

  Raises
  ------
  TypeError
      If a response is neither a Boolean nor an integer.
  ValueError
      If the responses and the administered items differ in number, or an integer response is not 0
      or 1.
  """
  checked = tuple(_response(f"responses[{position}]", value) for position, value in enumerate(values))
  if len(checked) != n_items:
    msg = (
      f"responses has {len(checked)} entries and administered_items {n_items}; there must be one response per "
      "administered item"
    )
    raise ValueError(msg)
  return checked


def _check_real(name: str, value: object, kind: str) -> None:
  """Check that a value is a real number, not a bool.

  Parameters
  ----------
  name : str
      The input the value is, for the message.
  value : object
      The value.
  kind : str
      What the value is, in the plural, for the message.

  Raises
  ------
  TypeError
      If the value is not a real number.
  """
  # math.isfinite("0.5") raises "must be real number, not str", which names no input.
  if isinstance(value, (bool, numpy.bool_)) or not isinstance(value, numbers.Real):
    msg = f"{name} is {value!r}; {kind} must be real numbers"
    raise TypeError(msg)


def _check_theta(name: str, value: float) -> None:
  """Check that an ability estimate is a finite number.

  Parameters
  ----------
  name : str
      The input the estimate is, for the message.
  value : float
      The estimate.

  Raises
  ------
  TypeError
      If the estimate is not a real number.
  ValueError
      If the estimate is NaN or infinite.
  """
  _check_real(name, value, "ability estimates")
  # A NaN estimate makes every item's information NaN, which changes the selected item.
  if not math.isfinite(value):
    msg = f"{name} is {value!r}; ability estimates must be finite"
    raise ValueError(msg)


def _check_se(name: str, value: float) -> None:
  """Check that a standard error is positive, or infinite.

  Parameters
  ----------
  name : str
      The input the standard error is, for the message.
  value : float
      The standard error.

  Raises
  ------
  TypeError
      If the standard error is not a real number.
  ValueError
      If the standard error is zero, negative or NaN.
  """
  _check_real(name, value, "standard errors")
  # A NaN error fails every comparison: a confidence-interval stopper would stop at once and a
  # standard-error stopper never.
  if not value > 0:
    msg = f"{name} is {value!r}; standard errors must be positive, or math.inf when the responses carry no information"
    raise ValueError(msg)


def _exposure_rates(rates: object, n_items: int) -> FloatArray | None:
  """Return exposure rates as a read-only float array, checking that there is one rate in [0, 1] per item.

  Parameters
  ----------
  rates : array-like or None
      Each item's exposure rate, in item bank order, or None, under which no exposure cap applies.
  n_items : int
      The number of items in the bank.

  Returns
  -------
  numpy.ndarray or None
      The rates, or None: ``rates`` itself if it is a read-only float array, and a read-only copy
      otherwise, so a strategy cannot change the caller's rates, or those of a later selection.

  Raises
  ------
  TypeError
      If the rates are not real numbers.
  ValueError
      If there is not one rate per item, or a rate is outside [0, 1].
  """
  if rates is None:
    return None
  array = numpy.asarray(rates)
  # Booleans, strings and objects are no rates: NumPy would read a mask as 0 and 1, and "0.5" as 0.5.
  if array.dtype.kind not in "iuf":
    msg = f"exposure_rates holds {array.dtype} values; exposure rates must be real numbers"
    raise TypeError(msg)
  # Rates of another length failed deep inside NumPy with an IndexError, or were broadcast.
  if array.shape != (n_items,):
    msg = f"exposure_rates has shape {array.shape}; there must be one exposure rate per item of the bank ({n_items})"
    raise ValueError(msg)
  if array.dtype != numpy.float64 or array.flags.writeable:
    array = array.astype(numpy.float64)
    array.flags.writeable = False
  # Exposure counts passed as rates were all above any cap below 1, which then held back no item.
  outside = numpy.flatnonzero(~((array >= 0.0) & (array <= 1.0)))
  if outside.size:
    position = int(outside[0])
    msg = f"exposure_rates[{position}] is {float(array[position])!r}; exposure rates must be between 0 and 1"
    raise ValueError(msg)
  return array
