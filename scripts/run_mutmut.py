"""Run mutmut with a macOS worker-process workaround.

mutmut currently has an open macOS issue where calling ``setproctitle()``
immediately after ``fork()`` can crash worker processes with ``SIGSEGV``.
This wrapper replaces that call with a no-op on Darwin before invoking the
normal mutmut CLI.
"""

from __future__ import annotations

import platform
import sys
from typing import TYPE_CHECKING

import mutmut.__main__ as mutmut_main

if TYPE_CHECKING:
  from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
  """Run mutmut, applying a macOS-only setproctitle workaround."""
  if platform.system() == "Darwin":
    mutmut_main.setproctitle = lambda _title: None

  args = list(argv) if argv is not None else sys.argv[1:]
  result = mutmut_main.cli.main(args=args, prog_name="mutmut", standalone_mode=False)
  return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
  raise SystemExit(main())
