"""catsim: Computerized Adaptive Testing engine and simulation toolkit."""

import importlib.metadata

from .context import Estimate, StepContext
from .engine import CatEngine, ResponseProvider, RunContext
from .exceptions import NoItemsAvailableError
from .item_bank import IrtModel, ItemBank
from .simulation import SimulatedResponseProvider, SimulationResult, SimulationRunner
from .state import CatSessionSnapshot, CatSessionState, CatStepResult, ExposureTracker, StopReason

try:  # noqa: RUF067 - the version lookup needs a fallback, so it cannot be a plain re-export
  __version__ = importlib.metadata.version("catsim")
except importlib.metadata.PackageNotFoundError:
  # Imported from a source tree that was never installed (e.g. PYTHONPATH=src): no metadata to read
  __version__ = "0+unknown"

__all__ = [
  "CatEngine",
  "CatSessionSnapshot",
  "CatSessionState",
  "CatStepResult",
  "Estimate",
  "ExposureTracker",
  "IrtModel",
  "ItemBank",
  "NoItemsAvailableError",
  "ResponseProvider",
  "RunContext",
  "SimulatedResponseProvider",
  "SimulationResult",
  "SimulationRunner",
  "StepContext",
  "StopReason",
]
