# AGENTS.md — /src/twinr/memory/longterm/runtime

## Scope

This directory owns runtime-facing orchestration for Twinr long-term memory:
service assembly, remote readiness probes, and bounded background writers.
Structural ownership and export metadata live in [component.yaml](./component.yaml).

Out of scope:
- extraction, retrieval, reflection, retention, or proactive algorithms themselves
- low-level object, graph, midterm, or remote-state storage implementations
- top-level agent runtime loops, hardware orchestration, or web flows

## Key files

- `service.py` — compatibility shim that preserves the historic import path and delegates to `service_impl/`
- `live_object_selectors.py` — shared query-first selector layer for live-near reserve/proactive/maintenance object reads
- `prepared_context.py` — bounded prepared provider/tool context front for transcript-first speculative warmup and hot-path reuse
- `context_snapshot.py` — latest built provider/tool context snapshot used by operator surfaces such as Conversation Lab
- `service_impl/compat.py` — shared helper functions, limits, logger, and readiness dataclasses extracted from the old monolith
- `service_impl/main.py` — `LongTermMemoryService` dataclass and inherited method surface
- `service_impl/builder.py` — service assembly and bounded writer construction
- `service_impl/readiness.py` — required-remote readiness probes and remote-state cache helpers
- `service_impl/context.py` — provider-context assembly for normal, fast, and tool-facing retrieval
- `service_impl/ingestion.py` — enqueue/import paths and dry-run analysis entry points
- `service_impl/maintenance.py` — reflection, sensor-memory, backfill, and retention orchestration
- `service_impl/proactive.py` — proactive planning and reservation state transitions
- `service_impl/mutations.py` — conflict resolution, review, and prompt-context mutation entry points
- `service_impl/lifecycle.py` — flush and shutdown orchestration for bounded background writers
- `service_impl/persistence.py` — static persistence helpers and multimodal/ops payload sanitization
- `worker.py` — bounded async persistence workers with exact drain/error tracking
- `flush_budget.py` — deterministic total-deadline planner for multi-writer flush orchestration
- `health.py` — remote snapshot readiness probe; no production state mutation here
- `component.yaml` — package metadata, callers, and test map

## Invariants

- `service.py` stays orchestration-focused. New extraction, reasoning, storage, or policy logic belongs in the package that already owns that concern.
- `service.py` stays a thin compatibility shim. New runtime logic belongs in the appropriate `service_impl/*.py` module instead of growing the shim again.
- Shared `_store_lock` serialization is reserved for object/graph/midterm mutation paths that truly share those stores. Prompt-context mutations keep their own store-local locking and must not wait behind unrelated background multimodal or turn persistence.
- Runtime provider/tool context assembly must consume `prepared_context.py` first and must not reintroduce duplicate synchronous full-context rebuilds when an identical speculative build is already in flight.
- Operator/debug surfaces may inspect the latest built provider/tool context snapshot, but they must not trigger a second independent remote recall merely to repaint the same turn context.
- Personality learning stays a downstream sidecar owned by `src/twinr/agent/personality/`; `service.py` may route consolidated turns and tool history into it, but must not reimplement signal taxonomy or evolution policy here.
- Foreground runtime turn-finalization paths must only queue tool-history learning and must not reacquire the shared long-term store lock for that queue step; any expensive remote-primary personality commit belongs to the later bounded persistence/flush path, not the answer-finalization hot path.
- When runtime reflection creates summaries or promotes objects inline, the downstream personality-learning handoff must see that enriched batch instead of only the pre-reflection consolidator result.
- Required remote-primary readiness failures must surface as `LongTermRemoteUnavailableError`; do not degrade them into empty context or silent fallback.
- `health.py` must attest graph and midterm readiness through the stores' current-view/current-head helpers, not by reintroducing generic `load_snapshot("graph"|"midterm")` checks.
- Background writers must stay bounded, reject new items after shutdown starts, and preserve exact pending/drop/error state.
- Service-level flush deadlines must be real wall-clock totals; do not reapply the full timeout independently to multiple writers.
- Runtime restart-recall persistence stays orchestration-only here: `service.py` may trigger packet refreshes, but packet compilation logic belongs in retrieval and packet storage semantics belong in storage.
- Live-near reserve/proactive/runtime maintenance callers must get object slices through `live_object_selectors.py`; do not add new direct `load_objects()` calls in runtime orchestration for those paths.
- Full-state runtime persistence and maintenance reads must come through the storage layer's shared fine-grained current-state loader; do not reintroduce direct snapshot blob hydration for required remote object/conflict/archive state in runtime orchestration.
- Conversation and multimodal dry-run analysis must use storage-owned active working sets; do not add new `load_objects()` blob hydration or broad full-state hydration to those runtime entry points.
- Reflection and sensor-memory runtime inputs should prefer bounded neighborhood selectors keyed from the touched objects or events instead of broad query unions whenever the caller has a concrete seed set available.
- Durable-memory writes that materially change live recall must invalidate the prepared context front so the next live answer cannot reuse stale provider/tool prompt artifacts.
- Runtime may persist a deterministic immediate turn-continuity packet before slower durable enrichment drains, but the packet-compilation logic belongs in reasoning and must not grow ad-hoc inside `service.py`.
- Multimodal evidence and episodic fallback payloads must remain bounded and sanitized before persistence.

## Verification

After any edit in this directory, run:

```bash
python3 -m compileall src/twinr/memory/longterm/runtime
PYTHONPATH=src pytest test/test_longterm_worker.py test/test_longterm_runtime_health.py test/test_longterm_runtime_query_selectors.py -q
```

If `service.py` changed, also run:

```bash
PYTHONPATH=src pytest test/test_longterm_memory.py test/test_longterm_multimodal.py test/test_longterm_proactive.py test/test_longterm_remote_state.py test/test_runtime_context.py test/test_longterm_midterm.py -q
```

If `prepared_context.py` or `service_impl/context.py` changed, also run:

```bash
PYTHONPATH=src pytest test/test_longterm_memory.py test/test_streaming_runner.py -q
```

## Coupling

`service.py` changes -> also check:
- `service_impl/`
- `src/twinr/memory/longterm/ingestion/`
- `src/twinr/memory/longterm/reasoning/`
- `src/twinr/memory/longterm/proactive/`
- `src/twinr/memory/longterm/storage/`
- `src/twinr/agent/personality/`
- `src/twinr/agent/base_agent/runtime/base.py`
- `test/test_longterm_memory.py`

`worker.py` changes -> also check:
- `service.py`
- `flush_budget.py`
- `test/test_longterm_worker.py`
- `test/test_longterm_memory.py`

`flush_budget.py` changes -> also check:
- `service.py`
- `worker.py`
- `test/test_longterm_memory.py`

`health.py` changes -> also check:
- `service.py`
- `src/twinr/agent/base_agent/runtime/base.py`
- `test/test_longterm_runtime_health.py`
- `test/test_longterm_remote_state.py`

`live_object_selectors.py` changes -> also check:
- `service_impl/proactive.py`
- `service_impl/maintenance.py`
- `service_impl/lifecycle.py`
- `src/twinr/memory/user_discovery_authoritative_profile.py`
- `src/twinr/proactive/runtime/display_reserve_reflection.py`
- `test/test_longterm_proactive.py`
- `test/test_user_discovery.py`
- `test/test_display_reserve_reflection.py`
- `test/test_longterm_runtime_query_selectors.py`

## Security

- Keep multimodal payload sanitization JSON-safe and size-bounded; do not persist raw arbitrary device payloads.
- Keep ops-history backfill restricted to validated regular files; do not widen path handling here.
- Do not mask required remote-primary failures behind best-effort fallback on readiness or startup paths.

## Output expectations

- Update docstrings when service entry points, queue semantics, or remote readiness behavior change.
- Keep [README.md](./README.md), [AGENTS.md](./AGENTS.md), and [component.yaml](./component.yaml) aligned when file roles, exports, or verification commands change.
- Treat export changes in `src/twinr/memory/longterm/__init__.py` and `src/twinr/memory/__init__.py` as API-impacting follow-up work.
