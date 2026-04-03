# runtime

Runtime-facing orchestration for Twinr long-term memory.

This package wires stores, reasoning helpers, remote readiness checks, and
bounded background writers into the APIs used by agent runtime loops.

## Responsibility

`runtime` owns:
- Assemble `LongTermMemoryService` from config and subsystem dependencies
- Build runtime provider context and tool-safe context while preserving both original-language and canonical query recall paths
- Let synchronous provider-context, prewarm, and explicit materialized-front builds wait a bounded first-turn window sized to real rewrite latency, so non-English graph/subtext cues are present on the first built front instead of only on a later cache refresh
- Maintain a remote-authoritative materialized answer-front for the live provider path so transcript-first turns can consume ChonkyDB-backed prompt sections instead of synchronously rebuilding the broad retriever package on every answer
- Seed that materialized live-provider front from compatibility `build_provider_context(...)` callers only through deduplicated background persistence, so non-live/eval provider builds do not block foreground answer/context assembly on remote current-head writes
- Keep the older in-process prepared full-context front only for tool context and compatibility provider callers while the live voice/runtime path migrates to the materialized answer-front contract
- Expose the latest built provider/tool context snapshot so operator surfaces can inspect the real turn context without launching a second independent remote recall
- Prime and refresh those prepared full-context fronts from transcript-first interim or endpoint transcripts plus post-write invalidations, so the first live answer can reuse already-started long-term retrieval work without shrinking the memory surface
- Persist those live provider answer fronts as bounded current-head collections keyed by semantic query scope, so post-write invalidation and transcript-first prewarm can keep remote-authoritative prompt blocks hot across turns
- Expose an explicit synchronous live-front materialization helper for non-streaming callers such as evals, so they can prime the strict live-provider contract without reintroducing a hidden synchronous fallback inside the live answer path
- Fail closed on raw graph-context blocks in the normal provider path when unified subtext planning found no relevant graph payload and no other recall sections survived, so graph-only control queries do not over-personalize answers with irrelevant person/preference context
- Reject new prepared-front or materialized-front background work after shutdown starts, so draining writers cannot poison same-process evals with post-shutdown executor submissions
- Install a cooperative shutdown signal around each in-flight background persistence item, so remote retry/backoff loops can abort promptly instead of holding service shutdown open on stale 429 cooldown sleeps
- Inject the bounded fast-topic quick-memory lane into the normal provider context so ordinary answer turns receive a few compact personalized topic hints before the heavier recall sections
- Build one bounded fast-topic provider context for direct/latency-sensitive reply lanes that only need a few current-topic hints before answering, while surfacing specific required remote-read failures separately from broader backend-unavailable state
- Expose confirmed durable-memory state explicitly in provider/tool context so meta-memory questions can distinguish stored current facts from generic neighbors
- Rebuild and persist provenance-rich restart-recall packets after durable-memory mutations so fresh runtime roots retain immediate continuity
- Centralize bounded query-first object selectors reused by discovery, proactive planning, reflection, sensor-memory compilation, and restart-recall refresh so live runtime paths do not hydrate full remote object snapshots or silently widen seeded neighborhood inputs back into broad compile-source unions
- Route conversation/multimodal persistence and backfill through storage-owned active working sets plus delta commits so required remote writes do not hydrate full object/conflict/archive state or rewrite whole snapshots before consolidation
- Prefilter remote-primary online retention candidates from current-head catalog projections before exact hydration, so retention no longer starts from a global active-object sweep
- Persist one deterministic immediate turn-continuity midterm packet before slower extraction/reflection drains so fresh follow-up recall does not block on the full background writer
- Persist conversation turns and multimodal evidence through bounded workers
- Preserve turn provenance such as external channel source and input modality so text channels like WhatsApp reach durable memory and personality learning as first-class turns instead of generic voice-only records
- Coalesce high-frequency sensor observations onto a latest-only idle-loop handoff before they hit multimodal long-term persistence, so proactive sensor churn cannot build an unbounded remote write backlog
- Route consolidated conversation turns and post-turn tool history into the structured personality-learning service without mixing that policy into the core memory algorithms
- Keep foreground turn finalization bounded by queueing tool-history learning without taking the shared long-term store lock first, then letting the later conversation-persistence/flush path commit those signals under the existing long-term store lock
- Pass reflection-enriched turn batches into personality learning so downstream continuity/context formation can see newly created thread summaries from the same committed turn
- Route explicit RSS/world-intelligence configuration plus reflection-phase feed refresh/recalibration into the structured personality-learning service without mixing that source logic into the core memory algorithms
- Split one service-level flush timeout into explicit per-writer budgets so wall-clock runtime deadlines stay real
- Keep low-signal multimodal writes on the deterministic reflection path while skipping optional midterm compilation that would otherwise duplicate per-turn remote latency
- Expose explicit delete entry points for durable prompt memory and managed user/personality context so discovery corrections can retract stale learned facts cleanly
- Verify remote-primary snapshot readiness before runtime loops start
- Record per-snapshot pointer/origin readiness evidence so watchdog failures can be traced to the exact remote read path
- Probe graph and midterm readiness through their store-owned current-view/current-head contracts so health attestation checks the same remote authority path used at runtime
- Keep graph/object/midterm bootstrap on dedicated `*_for_readiness` contracts so a fresh empty required namespace stays strictly read-only during startup instead of blocking on empty remote seed writes
- Let graph/midterm readiness fall back from a lagging direct fixed-URI head to the small loadable compatibility current-head contract, so fresh runtime roots stay fail-closed without reintroducing whole-snapshot blob reads or duplicate bootstrap writes
- Reuse successful remote snapshot probes within one bounded readiness pass so watchdog startup does not refetch the same snapshot twice back-to-back
- Propagate successful external watchdog attestations back into every owned remote-state adapter so stale local cooldown state cannot contradict the Pi's required-remote gate
- Treat successful required snapshot loads as the decisive health proof inside the warm probe instead of re-running per-store backend status checks after bootstrap
- Split remote readiness into a strict bootstrap pass and a no-bootstrap steady-state watchdog pass so live keepalives can reuse warm remote state without reseeding every snapshot on every tick
- Expose explicit attestation tiers so lighter current-only probes are visible as degraded/current-only instead of looking archive-safe
- Keep the external required-remote watchdog archive-inclusive even in steady state so a green watchdog sample remains archive-safe and valid for startup/recovery gating
- Fail closed when any required remote snapshot or shard is unreadable
- Prewarm generic foreground-read paths plus current object/conflict payload caches before live text-channel traffic so the first real remote-only recall turn can hit a warmed remote cache instead of rebuilding selectors and fetching first-hit payloads on demand
- Reuse remote snapshot/catalog/item reads through TTL-bounded in-process caches so warmed foreground recall stays sub-second on the Pi without changing remote-only truth semantics
- Keep foreground provider-context reads on the stores' own consistency boundaries instead of queueing them behind the shared background-writer mutation lock
- Keep prompt-context mutations on their own store-level durability boundary instead of queueing them behind the shared background-writer mutation lock
- Wire adaptive retrieval policies derived from confirmed memory, routines, and proactive outcomes into provider context
- Expose review, retention, proactive, and operator mutation entry points

`runtime` does **not** own:
- Define long-term memory schemas or ontology rules
- Implement extraction, consolidation, retrieval, reflection, or retention algorithms
- Implement storage backends, ChonkyDB transport, or remote snapshot formats
- Own top-level agent runtime loops or UI workflows

## Key files

| File | Purpose |
|---|---|
| `service.py` | Compatibility shim that preserves the historic import surface while delegating to `service_impl/` |
| `live_object_selectors.py` | Shared bounded query-first selector layer for live-near reserve/proactive/maintenance object reads, with seeded reflection/sensor neighborhoods kept strict instead of widening back to broad compile-source unions |
| `prepared_context.py` | Bounded prepared full-context front that stores completed provider/tool prompt artifacts, coalesces transcript-first speculative builds, and refreshes or invalidates them after durable-memory changes |
| `provider_answer_front.py` | Runtime manager for remote-backed materialized live provider answer fronts, with transcript-first prewarm coalescing, local hot-cache reuse, and strict live consumption |
| `context_snapshot.py` | Small runtime dataclass that captures the latest built provider/tool context for operator inspection |
| `service_impl/README.md` | Internal map of the refactored service implementation package |
| `service_impl/main.py` | `LongTermMemoryService` dataclass plus inherited runtime method surface, including both the compatibility prepared-context front and the materialized live provider-front dependency |
| `service_impl/builder.py` | Runtime service assembly and background-writer construction, plus prepared-context/materialized-front wiring and persistence invalidation hooks |
| `service_impl/readiness.py` | Required-remote readiness probes and cache helpers |
| `service_impl/context.py` | Provider-context assembly for normal, live, fast, and tool-facing paths, with bounded first-rewrite waits for sync/prewarm materialization, live materialized-front consumption, compatibility prepared-front fallback, transcript-first prewarm entry points, sticky-query refresh scheduling, and fail-closed raw-graph suppression when no relevant graph payload survived |
| `service_impl/ingestion.py` | Conversation/multimodal enqueue and dry-run analysis entry points, with dry-run object reads sourced through storage-owned active working sets instead of broad current-state hydration |
| `service_impl/maintenance.py` | Reflection, sensor-memory, backfill, and retention orchestration, with reflection/sensor inputs narrowed through neighborhood selectors, active backfill writes using storage-owned working-set delta commits, and remote-primary retention prefiltering candidates from catalog projections |
| `service_impl/proactive.py` | Proactive planning and reservation state transitions using shared query-first planner-object selectors instead of inline query constants |
| `service_impl/mutations.py` | Conflict resolution, review, and prompt-context mutation entry points |
| `service_impl/lifecycle.py` | Flush/shutdown orchestration for bounded background writers plus query-first restart-recall packet refresh |
| `service_impl/persistence.py` | Static persistence pipeline helpers and multimodal payload sanitization, with conversation/multimodal writes using storage-owned working-set delta commits |
| `service_impl/compat.py` | Shared limits, logger, helper functions, and remote-readiness dataclasses |
| `worker.py` | Bounded async persistence |
| `flush_budget.py` | Total-deadline planner for multi-writer flush budgeting |
| `health.py` | Remote readiness probe with per-snapshot pointer/origin evidence |
| `component.yaml` | Structured package metadata |
| `AGENTS.md` | Local editing rules |

No migration is required for callers that already import
`twinr.memory.longterm.runtime.service`.

## Usage

```python
from twinr.memory.longterm import LongTermMemoryService

service = LongTermMemoryService.from_config(config)
service.ensure_remote_ready()
steady_state = service.probe_remote_ready(bootstrap=False, include_archive=False)
service.prewarm_provider_context("Wie ist Janinas Termin heute?")
service.materialize_live_provider_context("Soll ich Corinna heute noch anrufen?")
context = service.build_provider_context(query_text)  # includes quick-memory topic hints when enabled
fast_context = service.build_fast_provider_context(query_text)  # compact quick-memory-only context
service.enqueue_conversation_turn(
    transcript=user_text,
    response=assistant_text,
    source="whatsapp",
    modality="text",
)
service.enqueue_personality_tool_history(
    tool_calls=tool_calls,
    tool_results=tool_results,
)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [../MEMORY_ARCHITECTURE.md](../MEMORY_ARCHITECTURE.md)
- [../ingestion/README.md](../ingestion/README.md)
- [../reasoning/README.md](../reasoning/README.md)
- [../storage/README.md](../storage/README.md)
