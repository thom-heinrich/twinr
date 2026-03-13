"""
Contract
- Purpose: Provide a structured bugfix memory tool (FixReport) for agents.
- Inputs (types, units): Structured vocab fields (tokens/enums) + optional narrative strings.
- Outputs (types, units): YAML files under artifacts/stores/fixreport/; JSON-only CLI stdout.
- Invariants: Each report has unique bf_id (BF######) and valid vocab fields.
- Error semantics: Fail-fast with structured JSON error; never silently drops fields.
- Time/Horizon: All timestamps are ISO-8601 UTC with Z suffix; optional `horizon` is enum.
- External boundaries: Filesystem only (artifacts/stores/fixreport/), optional git introspection for commit SHA.
- Telemetry (metrics + log keys): stderr logs; JSON has {ok,error,detail}.
- Performance notes: Search uses index.yml when available; falls back to scanning reports.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
