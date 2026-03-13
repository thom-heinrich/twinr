"""
Contract
- Purpose:
  - Consistent logging for healthstream operations (stderr).
- Inputs:
  - Log level string.
- Outputs:
  - Configured stdlib logging.
- Invariants:
  - Logs include `healthstream` marker for grepping.
- Error semantics:
  - Never raises; safe to call multiple times.
"""

from __future__ import annotations

import logging
import sys
import time


def setup_logging(level: str) -> None:
    lvl = (level or "INFO").upper()
    logging.basicConfig(
        level=lvl,
        format="%(asctime)sZ %(levelname)s healthstream %(message)s",
        stream=sys.stderr,
    )
    logging.Formatter.converter = time.gmtime  # type: ignore[attr-defined]

