"""Type aliases for NumPy arrays, shared by every catsim module.

Annotate arrays with these aliases rather than spelling out ``numpy.typing.NDArray``. Ruff bans
``NDArray`` everywhere else (rule TID251), so a dtype without an alias here gets a new one.
"""

from typing import Any

import numpy
import numpy.typing as npt

FloatArray = npt.NDArray[numpy.floating[Any]]
"""A NumPy array of floating-point values of any precision."""

IntArray = npt.NDArray[numpy.integer[Any]]
"""A NumPy array of integers of any precision, such as item indices or exposure counts."""
