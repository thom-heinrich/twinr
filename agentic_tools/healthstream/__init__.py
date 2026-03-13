"""
Contract
- Purpose:
  - Public API surface for the `healthstream` agentic tool (central health/DQ/ops event stream).
- Inputs (types, units):
  - See `agentic_tools.healthstream.types.HealthStreamEvent`.
- Outputs (types, units):
  - Store read/write helpers and a CLI entrypoint (`agentic_tools.healthstream.cli`).
- Invariants:
  - UTC timestamps (ISO8601 with Z).
  - JSON-only store on disk with file locking and atomic writes.
  - No external dependencies (stdlib only).
- Error semantics:
  - Library functions raise `ValueError` / `RuntimeError` with details.
  - CLI emits JSON `{ok:false,...}` on failure and exits non-zero.
- Time/Horizon:
  - UTC-only; no horizon semantics.
- External boundaries:
  - Filesystem only (store file + lock file).
- Telemetry (metrics + log keys):
  - Logs to stderr include `tool=healthstream` and operation keys.
"""

from __future__ import annotations

from agentic_tools.healthstream.store import emit_event, get_store_path, list_events
from agentic_tools.healthstream.types import HealthStreamEvent

__all__ = [
    "HealthStreamEvent",
    "emit_event",
    "get_store_path",
    "list_events",
]

