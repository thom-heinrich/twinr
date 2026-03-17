# AGENTS.md — /src/twinr

## Scope

This directory owns the root `twinr` package boundary: the lazy top-level
import surface, the `python -m twinr` bootstrap, the root compatibility
re-export modules, and the small shared helper modules reused across multiple
subsystems. Structural ownership and entrypoints live in
[component.yaml](./component.yaml).

Out of scope:
- deep runtime, tool, and workflow behavior in `src/twinr/agent`
- provider implementations in `src/twinr/providers`
- hardware adapters in `src/twinr/hardware`
- memory internals in `src/twinr/memory`
- dashboard routes, presenters, and stores in `src/twinr/web`

## Key files

- `__init__.py` — lazy `twinr` export surface; treat export changes as API-impacting
- `__main__.py` — package CLI bootstrap and command dispatch
- `llm_json.py` — structured JSON request and fallback schema validation helper
- `temporal.py` — timezone-aware local date parser
- `text_utils.py` — shared text, JSON, and identifier normalization helper
- `config.py` / `runtime.py` / `runner.py` / `realtime_runner.py` / `state_machine.py` / `contracts.py` / `personality.py` — compatibility re-export surface

## Invariants

- Keep `src/twinr` thin. New product behavior belongs in a focused child package unless it is truly package-boundary, bootstrap, or cross-subsystem utility code.
- `__main__.py` may parse args, guard runtime preconditions, and dispatch commands, but business logic must stay in imported subsystem modules.
- `__init__.py` and the root compatibility modules must stay import-light and side-effect free; do not add eager runtime setup there.
- `llm_json.py`, `temporal.py`, and `text_utils.py` must stay generic shared helpers, not a dumping ground for subsystem-specific policy.
- When ownership changes between the root package and a child package, update the touched module docstrings plus [README.md](./README.md), [AGENTS.md](./AGENTS.md), and [component.yaml](./component.yaml) in the same change.

## Verification

After any edit in this directory, run:

```bash
python3 -m py_compile src/twinr/*.py
```

If `__init__.py`, `__main__.py`, or any root compatibility module changed, also run:

```bash
PYTHONPATH=src pytest test/test_main.py test/test_config.py test/test_runner.py test/test_realtime_runner.py test/test_streaming_runner.py -q
```

If `llm_json.py`, `temporal.py`, or `text_utils.py` changed, also run:

```bash
PYTHONPATH=src pytest test/test_llm_json.py test/test_temporal.py -q
```

## Coupling

`__main__.py` changes → also check:
- `src/twinr/agent/workflows/`
- `src/twinr/ops/`
- `src/twinr/providers/`
- `src/twinr/orchestrator/`
- `src/twinr/web/`
- `test/test_main.py`

`__init__.py` or root compatibility-module changes → also check:
- `src/twinr/agent/__init__.py`
- `src/twinr/memory/__init__.py`
- importers under `src/twinr/` and `test/`
- `test/test_config.py`
- `test/test_runner.py`
- `test/test_realtime_runner.py`

`llm_json.py` changes → also check:
- `src/twinr/memory/query_normalization.py`
- `src/twinr/memory/longterm/ingestion/propositions.py`
- `src/twinr/memory/longterm/reasoning/midterm.py`
- `src/twinr/memory/longterm/retrieval/subtext.py`
- `test/test_llm_json.py`

`temporal.py` changes → also check:
- `src/twinr/memory/chonkydb/personal_graph.py`
- `test/test_temporal.py`

`text_utils.py` changes → also check:
- `src/twinr/memory/context_store.py`
- `src/twinr/memory/fulltext.py`
- `src/twinr/memory/query_normalization.py`
- `src/twinr/memory/longterm/ingestion/extract.py`
- `src/twinr/memory/longterm/retrieval/retriever.py`
- `src/twinr/proactive/wakeword/matching.py`

## Output expectations

- Update in-script docstrings whenever root-module behavior or exports change.
- Prefer moving subsystem-specific code into a documented child package over extending a root helper.
- Keep root-package docs about boundaries and entrypoints only; deeper behavior belongs in the relevant child package docs.
