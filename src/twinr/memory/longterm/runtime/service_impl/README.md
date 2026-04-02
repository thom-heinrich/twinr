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
| `builder.py` | `from_config` dependency wiring, writer construction, and prepared-context/materialized-provider-front invalidation plumbing |
| `readiness.py` | Required-remote readiness probes and remote-state cache helpers, with dedicated graph/object/midterm `*_for_readiness` bootstrap contracts so fresh empty namespaces stay read-only during startup |
| `context.py` | Normal, live, fast, and tool-facing provider-context assembly, including strict materialized live-provider-front consumption, compatibility prepared full-context fallback, transcript-first speculative prewarm entry points, asynchronous seeding of the materialized live-provider front from compatibility provider builds, remote-memory completion stamps for the live voice-turn latency breakdown, and single-pass tool-context retrieval without duplicate remote durable/conflict reads while keeping graph loading as a fallback only when structured tool recall is empty |
| `ingestion.py` | Conversation/multimodal enqueue paths and dry-run analysis helpers, with dry-run object reads sourced through storage-owned active working sets instead of `load_objects()` blob hydration or broad current-state hydration |
| `maintenance.py` | Reflection, sensor-memory, backfill, and retention orchestration, with reflection/sensor inputs sourced through the shared runtime selector layer plus strict seeded neighborhood expansion, backfill writes using storage-owned active working sets plus delta commits instead of full-state rewrites, and remote-primary retention prefiltering candidate objects from catalog projections before exact hydration |
| `proactive.py` | Proactive planning, reservation, and outcome tracking over the shared query-first planner selector layer instead of inline query constants |
| `mutations.py` | Conflict resolution, review, and prompt-context mutation entry points, with conflict/object reads narrowed through store-level exact-id/slot-key/reference selectors and mutation writes committed as touched deltas instead of global conflict sweeps or full snapshot rewrites |
| `lifecycle.py` | Flush and shutdown orchestration for bounded background writers plus query-first restart-recall refresh |
| `persistence.py` | Static persistence pipeline helpers and multimodal/ops payload sanitization, with conversation/multimodal writes using storage-owned active working sets plus delta commits instead of full-state rewrites |
