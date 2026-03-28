# service_impl

Internal implementation package for [`../service.py`](../service.py).

`service.py` remains the stable import surface and compatibility shim.
No caller migration is required.

## Layout

| File | Purpose |
|---|---|
| `compat.py` | Shared helper functions, limits, logger, and remote-readiness dataclasses |
| `main.py` | `LongTermMemoryService` dataclass and inherited method surface |
| `builder.py` | `from_config` dependency wiring and writer construction |
| `readiness.py` | Required-remote readiness probes and remote-state cache helpers |
| `context.py` | Normal, fast, and tool-facing provider-context assembly |
| `ingestion.py` | Conversation/multimodal enqueue paths and dry-run analysis helpers |
| `maintenance.py` | Reflection, sensor-memory, backfill, and retention orchestration |
| `proactive.py` | Proactive planning, reservation, and outcome tracking |
| `mutations.py` | Conflict resolution, review, and prompt-context mutation entry points |
| `lifecycle.py` | Flush and shutdown orchestration for bounded background writers |
| `persistence.py` | Static persistence pipeline helpers and multimodal/ops payload sanitization |
