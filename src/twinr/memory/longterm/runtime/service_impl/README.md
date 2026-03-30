# service_impl

Internal implementation package for [`../service.py`](../service.py).

`service.py` remains the stable import surface and compatibility shim.
No caller migration is required. Live-near object selection is centralized in
[`../live_object_selectors.py`](../live_object_selectors.py) so the mixins
below do not each own ad-hoc query strings.

## Layout

| File | Purpose |
|---|---|
| `compat.py` | Shared helper functions, limits, logger, and remote-readiness dataclasses |
| `main.py` | `LongTermMemoryService` dataclass and inherited method surface |
| `builder.py` | `from_config` dependency wiring and writer construction |
| `readiness.py` | Required-remote readiness probes and remote-state cache helpers |
| `context.py` | Normal, fast, and tool-facing provider-context assembly |
| `ingestion.py` | Conversation/multimodal enqueue paths and dry-run analysis helpers, with dry-run object reads sourced through storage fine-grained current-state loaders instead of `load_objects()` blob hydration |
| `maintenance.py` | Reflection, sensor-memory, backfill, and retention orchestration, with reflection/sensor inputs sourced through the shared runtime selector layer, backfill writes using storage-owned active working sets plus delta commits instead of full-state rewrites, and remote-primary retention prefiltering candidate objects from catalog projections before exact hydration |
| `proactive.py` | Proactive planning, reservation, and outcome tracking over the shared query-first planner selector layer instead of inline query constants |
| `mutations.py` | Conflict resolution, review, and prompt-context mutation entry points, with conflict/object reads narrowed through store-level exact-id/slot-key/reference selectors plus fine-grained current-state fallback instead of full snapshot hydration |
| `lifecycle.py` | Flush and shutdown orchestration for bounded background writers plus query-first restart-recall refresh |
| `persistence.py` | Static persistence pipeline helpers and multimodal/ops payload sanitization, with conversation/multimodal writes using storage-owned active working sets plus delta commits instead of full-state rewrites |
