"""catsim: Computerized Adaptive Testing engine and simulation toolkit."""

import importlib.metadata

from .context import Estimate, StepContext
from .engine import CatEngine, Selection
from .exceptions import ExcludedItemError, NoItemsAvailableError
from .session import Session, StopReason
from .simulation import SimulationResult, simulate

try:  # noqa: RUF067 - the version lookup needs a fallback, so it cannot be a plain re-export
  __version__ = importlib.metadata.version("catsim")
except importlib.metadata.PackageNotFoundError:
  # Imported from a source tree that was never installed (e.g. PYTHONPATH=src): no metadata to read
  __version__ = "0+unknown"

__all__ = [
  "CatEngine",
  "Estimate",
  "ExcludedItemError",
  "NoItemsAvailableError",
  "Selection",
  "Session",
  "SimulationResult",
  "StepContext",
  "StopReason",
  "simulate",
]
