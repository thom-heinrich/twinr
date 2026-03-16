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

- `service.py` — runtime orchestration entrypoint; keep deep business logic in owned subpackages
- `worker.py` — bounded async persistence workers with exact drain/error tracking
- `health.py` — remote snapshot readiness probe; no production state mutation here
- `component.yaml` — package metadata, callers, and test map

## Invariants

- `service.py` stays orchestration-focused. New extraction, reasoning, storage, or policy logic belongs in the package that already owns that concern.
- All runtime mutations that touch prompt, object, graph, or midterm stores must stay serialized under the shared `_store_lock`.
- Required remote-primary readiness failures must surface as `LongTermRemoteUnavailableError`; do not degrade them into empty context or silent fallback.
- Background writers must stay bounded, reject new items after shutdown starts, and preserve exact pending/drop/error state.
- Multimodal evidence and episodic fallback payloads must remain bounded and sanitized before persistence.

## Verification

After any edit in this directory, run:

```bash
python3 -m compileall src/twinr/memory/longterm/runtime
PYTHONPATH=src pytest test/test_longterm_worker.py test/test_longterm_runtime_health.py -q
```

If `service.py` changed, also run:

```bash
PYTHONPATH=src pytest test/test_longterm_memory.py test/test_longterm_multimodal.py test/test_longterm_proactive.py test/test_longterm_remote_state.py -q
```

## Coupling

`service.py` changes -> also check:
- `src/twinr/memory/longterm/ingestion/`
- `src/twinr/memory/longterm/reasoning/`
- `src/twinr/memory/longterm/proactive/`
- `src/twinr/memory/longterm/storage/`
- `src/twinr/agent/base_agent/runtime/base.py`
- `test/test_longterm_memory.py`

`worker.py` changes -> also check:
- `service.py`
- `test/test_longterm_worker.py`
- `test/test_longterm_memory.py`

`health.py` changes -> also check:
- `service.py`
- `src/twinr/agent/base_agent/runtime/base.py`
- `test/test_longterm_runtime_health.py`
- `test/test_longterm_remote_state.py`

## Security

- Keep multimodal payload sanitization JSON-safe and size-bounded; do not persist raw arbitrary device payloads.
- Keep ops-history backfill restricted to validated regular files; do not widen path handling here.
- Do not mask required remote-primary failures behind best-effort fallback on readiness or startup paths.

## Output expectations

- Update docstrings when service entry points, queue semantics, or remote readiness behavior change.
- Keep [README.md](./README.md), [AGENTS.md](./AGENTS.md), and [component.yaml](./component.yaml) aligned when file roles, exports, or verification commands change.
- Treat export changes in `src/twinr/memory/longterm/__init__.py` and `src/twinr/memory/__init__.py` as API-impacting follow-up work.
