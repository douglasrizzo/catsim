"""Initialization components for CAT."""

from .base import BaseInitializer
from .initialization import FixedPointInitializer, InitializationDistribution, RandomInitializer

# Backward compatibility alias
Initializer = BaseInitializer

__all__ = [
  "BaseInitializer",
  "FixedPointInitializer",
  "InitializationDistribution",
  "Initializer",  # Backward compatibility alias
  "RandomInitializer",
]
