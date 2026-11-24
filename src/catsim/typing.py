"""Type aliases for NumPy arrays, used in catsim's public signatures.

The aliases are public, so code that extends catsim, such as a custom selector, can annotate its
arrays the way catsim does. Inside catsim, Ruff bans ``numpy.typing.NDArray`` everywhere but here
(rule TID251), so a dtype without an alias here gets a new one.
"""

from typing import Any

import numpy
import numpy.typing as npt

FloatArray = npt.NDArray[numpy.floating[Any]]
"""A NumPy array of floating-point values of any precision."""

IntArray = npt.NDArray[numpy.integer[Any]]
"""A NumPy array of integers of any precision, such as item indices or exposure counts."""
