"""
Findings tool (agentic CLI)

Findings are *pre-fix* issue notes that can later be converted into Tasks and/or
resolved via FixReports. They are designed to participate in the repo-wide
cross-tool meta graph via explicit `links: ["kind:id", ...]` tokens.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"

