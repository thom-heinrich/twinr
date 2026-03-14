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
  - immutable runtime and versioned long-term memory object models
- `worker.py`
  - bounded background writer thread so durable turn storage does not block the response path
- `service.py`
  - one orchestration surface for retrieval, explicit memory writes, and background episodic persistence
- `retriever.py`
  - hybrid retrieval and bounded context assembly for silent personalization, explicit recall, and conflict prompts
- `extract.py`
  - cautious turn-to-memory decomposition into episode, fact, event, observation, and graph-edge candidates
- `multimodal.py`
  - cautious device-event decomposition for PIR, camera, button, and printer signals into low-confidence multimodal memory candidates
- `truth.py`
  - truth-maintenance primitives for slot conflicts, activation, and clarification gating
- `consolidator.py`
  - promotion of extracted candidates into episodic vs durable memory plus conflict-aware graph emission
- `store.py`
  - local structured persistence for durable objects and unresolved conflicts, plus review/delete/invalidate mutation helpers
- `reflect.py`
  - bounded reflection over stored objects for support-count promotion and compact thread summaries
- `planner.py`
  - bounded proactive candidate generation from durable and reflected long-term memory
- `proactive.py`
  - configurable reservation, cooldown, sensitivity gating, and delivery-history state for live long-term proactive prompts
- `conflicts.py`
  - queued clarification items plus explicit conflict resolution that updates object validity instead of silently drifting
- `retention.py`
  - bounded retention policy for expiring time-bound memory and pruning low-value ephemeral objects
- `subtext.py`
  - builds silent personalization cues that should shape replies without explicit memory announcements
- `subtext_eval.py`
  - bounded live reply eval for subtle personalization quality
- `multimodal_eval.py`
  - deterministic seeded multimodal goldset for provider-context retrieval quality over PIR, camera, button, printer, and episodic context
- `../query_normalization.py`
  - canonical retrieval query rewriting plus low-information token filtering for multilingual memory lookup

## Current runtime behavior

When enabled, Twinr uses the long-term service in two directions:

1. Before a provider call, it builds a structured long-term context package:
   - silent personalization subtext
   - recent episodic memories
   - graph-based personalization context
   - conflict clarification context when relevant
2. After a normal conversation turn finishes, it can queue an episodic memory write in the background.
3. During live operation, it can also queue multimodal evidence from:
   - PIR/camera sensor observations
   - green/yellow button usage
   - completed print deliveries
   - live camera captures
4. When enabled, the background loops can reserve one bounded long-term proactive candidate at a time, apply cooldown/sensitivity policy, and speak it through the existing proactive prompt path.

The service keeps semantic memory text in canonical English. Names, phone numbers, email addresses, IDs, and verbatim quoted text stay unchanged.

For non-English user queries, the retrieval path can build a short canonical English retrieval query before ranking memories. This keeps memory content English without relying on brittle per-language prompt rules in the prompt builder.

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
- `TWINR_LONG_TERM_MEMORY_PROACTIVE_ENABLED`
- `TWINR_LONG_TERM_MEMORY_PROACTIVE_POLL_INTERVAL_S`
- `TWINR_LONG_TERM_MEMORY_PROACTIVE_MIN_CONFIDENCE`
- `TWINR_LONG_TERM_MEMORY_PROACTIVE_REPEAT_COOLDOWN_S`
- `TWINR_LONG_TERM_MEMORY_PROACTIVE_SKIP_COOLDOWN_S`
- `TWINR_LONG_TERM_MEMORY_PROACTIVE_RESERVATION_TTL_S`
- `TWINR_LONG_TERM_MEMORY_PROACTIVE_ALLOW_SENSITIVE`
- `TWINR_LONG_TERM_MEMORY_PROACTIVE_HISTORY_LIMIT`
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

## Multimodal goldset eval

Twinr also carries a deterministic multimodal long-term eval under `src/twinr/memory/longterm/multimodal_eval.py`.

This harness seeds a realistic mixed store with:

- 200 multimodal device events
- 300 episodic conversation turns
- 50 fixed goldset cases over presence, button/print routine, camera interaction, combined episodic+multimodal context, and irrelevant-query controls

It exercises the full provider-context path with canonical-English retrieval queries, rather than checking only raw extractor outputs.

Repro:

```bash
PYTHONPATH=src ./.venv/bin/python -m twinr.memory.longterm.multimodal_eval
```

Current validated run on 2026-03-14:

- 500 seeded entries
- 50/50 cases passed
- 1.0 accuracy

This eval also guards against two subtle quality regressions:

- internal machine metadata like `slot_key` or `support_count` leaking into retrieval ranking
- irrelevant episodic fallback context appearing on unrelated user questions

## Structured persistence

The long-term service now keeps a local structured store alongside the markdown episodic path.

- structured memory objects:
  - `TWINR_LONG_TERM_MEMORY_PATH/twinr_memory_objects_v1.json`
- unresolved conflicts:
  - `TWINR_LONG_TERM_MEMORY_PATH/twinr_memory_conflicts_v1.json`

The async background path can now:

1. persist the old markdown episodic turn
2. extract structured memory candidates from the same turn
3. consolidate them against existing structured memory
4. persist durable objects and unresolved conflicts for later retrieval
5. run a bounded reflection pass to strengthen repeated evidence and emit `thread_summary` objects
6. expose a bounded proactive planner over structured memory for reminders and gentle follow-ups
7. apply bounded retention so old episodes/observations do not accumulate without bound and past events become `expired`

The long-term service can now also surface queued clarification conflicts with concrete answer options and apply explicit resolutions. When a user confirms which memory is correct, Twinr keeps provenance and updates the competing memory states instead of silently overwriting them.

Alongside conflict resolution, the long-term layer now also supports direct review and manual memory hygiene in the same structured store:

- review durable memory objects without dragging unrelated episodic noise into the result set
- confirm a memory as user-validated
- invalidate a memory without deleting provenance
- delete a memory and clean dangling conflict/reference fields

These are currently memory-layer APIs on `LongTermMemoryService` / `LongTermStructuredStore`, so they are available for future portal and voice-tool wiring without inventing a second correction path.

The realtime voice tool path now exposes this explicitly through:

- `get_memory_conflicts`
  - inspect queued conflict questions and option IDs
- `resolve_memory_conflict`
  - activate the chosen option for a slot after spoken clarification

This keeps spoken clarification generic across contacts, preferences, routines, and other slot-based long-term facts instead of hard-coding one-off disambiguation cases.

Multimodal evidence uses the same consolidation/reflection/retention path as conversation turns, but starts from lower-confidence pattern candidates. This lets repeated PIR, camera, button, and printer signals strengthen into durable memory without over-trusting one isolated device event.

Live long-term proactive prompts now sit behind a separate reservation/policy layer instead of letting the planner speak directly. That layer is responsible for:

- minimum confidence gating
- sensitivity blocking unless explicitly enabled
- repeat cooldown after successful delivery
- skip cooldown after recent failure/suppression
- short reservation TTL so a stale reserved candidate does not loop forever after crashes
- small durable history under `TWINR_LONG_TERM_MEMORY_PATH/twinr_memory_proactive_state_v1.json`

This keeps the planner pure and keeps live delivery behavior configurable without coupling policy to retrieval or extraction.
