# Long-Term Memory Architecture

Twinr's long-term memory runtime path now lives under `src/twinr/memory/longterm/`.

The current target-state architecture for a frontier-grade Twinr memory system is documented separately in:

- `src/twinr/memory/longterm/MEMORY_ARCHITECTURE.md`

This layer keeps long-term memory responsibilities separated from:

- `src/twinr/memory/on_device/` for short rolling conversation memory
- `src/twinr/memory/context_store.py` for durable markdown/profile stores
- `src/twinr/memory/chonkydb/` for ChonkyDB client, graph schema, and personal graph logic
- `src/twinr/agent/base_agent/runtime.py` for turn orchestration

## Module split

- `models.py`
  - small immutable data models for queued conversation turns and retrieved long-term context
- `worker.py`
  - bounded background writer thread so durable turn storage does not block the response path
- `service.py`
  - one orchestration surface for retrieval, explicit memory writes, and background episodic persistence
- `subtext.py`
  - builds silent personalization cues that should shape replies without explicit memory announcements
- `subtext_eval.py`
  - bounded live reply eval for subtle personalization quality
- `../query_normalization.py`
  - canonical retrieval query rewriting plus low-information token filtering for multilingual memory lookup

## Current runtime behavior

When enabled, Twinr uses the long-term service in two directions:

1. Before a provider call, it builds a structured long-term context package:
   - silent personalization subtext
   - recent episodic memories
   - graph-based personalization context
2. After a normal conversation turn finishes, it can queue an episodic memory write in the background.

The service keeps semantic memory text in canonical English. Names, phone numbers, email addresses, IDs, and verbatim quoted text stay unchanged.

For non-English user queries, the retrieval path can build a short canonical English retrieval query before ranking memories. This keeps memory content English without relying on brittle per-language heuristics in the prompt builder.

## Current storage split

- recent episodic long-term memories:
  - `state/MEMORY.md`
- local graph-backed long-term structures:
  - `TWINR_LONG_TERM_MEMORY_PATH/twinr_graph_v1.json`

The normal live runtime still does not push every turn into the shared external ChonkyDB service. The external client remains available for future remote retrieval/write integration without coupling the current assistant loop to network latency.

## Config

- `TWINR_LONG_TERM_MEMORY_ENABLED`
- `TWINR_LONG_TERM_MEMORY_BACKEND`
- `TWINR_LONG_TERM_MEMORY_PATH`
- `TWINR_LONG_TERM_MEMORY_BACKGROUND_STORE_TURNS`
- `TWINR_LONG_TERM_MEMORY_WRITE_QUEUE_SIZE`
- `TWINR_LONG_TERM_MEMORY_RECALL_LIMIT`
- `TWINR_LONG_TERM_MEMORY_QUERY_REWRITE_ENABLED`
- `TWINR_CHONKYDB_BASE_URL`
- `TWINR_CHONKYDB_API_KEY`
- `TWINR_CHONKYDB_API_KEY_HEADER`
- `TWINR_CHONKYDB_ALLOW_BEARER_AUTH`
- `TWINR_CHONKYDB_TIMEOUT_S`

## Design constraints

- retrieval must stay deterministic and small
- response generation must not block on background memory persistence
- persistent writes must fail clearly instead of silently mutating state
- memory context should influence tone and personalization only when relevant
- hidden memory should usually shape replies silently instead of being announced explicitly
- the user-facing reply language stays independent from the internal memory language

By default, the long-term path should stay project-local under `state/chonkydb` so temporary runs and secondary env files do not leak graph data into a shared `/twinr` tree.

## Synthetic eval

Twinr now also carries a deterministic synthetic long-term memory eval harness under `src/twinr/memory/longterm/eval.py`.

The current harness seeds:

- 500 synthetic memories
- 50 dialogue/eval cases

Covered case families:

- exact contact recall
- contact disambiguation
- shopping recall
- temporal multihop recall
- episodic conversation recall

Repro:

```bash
PYTHONPATH=src ./.venv/bin/python -m twinr.memory.longterm.eval
```

## Response-level subtext eval

Twinr also carries a bounded live response eval for the silent personalization layer under `src/twinr/memory/longterm/subtext_eval.py`.

This eval uses real generated replies and checks whether hidden personal context is woven into answers naturally instead of being announced explicitly.
It runs with isolated per-case base instructions so unrelated repo personality or user files do not pollute the measurement.

Current case families:

- preference-based personalization
- situational continuity
- social-role familiarity
- episodic continuity
- irrelevant-query controls

Repro:

```bash
PYTHONPATH=src ./.venv/bin/python -m twinr.memory.longterm.subtext_eval
```

Current validated run on 2026-03-14:

- 8/8 passed
- 0 explicit memory-announcement violations
- average naturalness score: 4.75/5
