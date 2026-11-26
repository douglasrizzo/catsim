"""Standard-library names that catsim's oldest supported Python lacks, taken there from their backports.

catsim depends on typing-extensions only below Python 3.12 (ADR-0021), so this is the one module
that imports it, and only on those versions.
"""

import sys

if sys.version_info >= (3, 12):
  from typing import override
else:
  from typing_extensions import override

__all__ = ["override"]
