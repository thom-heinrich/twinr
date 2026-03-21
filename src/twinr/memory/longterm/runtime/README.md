# runtime

Runtime-facing orchestration for Twinr long-term memory.

This package wires stores, reasoning helpers, remote readiness checks, and
bounded background writers into the APIs used by agent runtime loops.

## Responsibility

`runtime` owns:
- Assemble `LongTermMemoryService` from config and subsystem dependencies
- Build runtime provider context and tool-safe context while preserving both original-language and canonical query recall paths
- Expose confirmed durable-memory state explicitly in provider/tool context so meta-memory questions can distinguish stored current facts from generic neighbors
- Rebuild and persist provenance-rich restart-recall packets after durable-memory mutations so fresh runtime roots retain immediate continuity
- Persist one deterministic immediate turn-continuity midterm packet before slower extraction/reflection drains so fresh follow-up recall does not block on the full background writer
- Persist conversation turns and multimodal evidence through bounded workers
- Coalesce high-frequency sensor observations onto a latest-only idle-loop handoff before they hit multimodal long-term persistence, so proactive sensor churn cannot build an unbounded remote write backlog
- Route consolidated conversation turns and post-turn tool history into the structured personality-learning service without mixing that policy into the core memory algorithms
- Pass reflection-enriched turn batches into personality learning so downstream continuity/context formation can see newly created thread summaries from the same committed turn
- Route explicit RSS/world-intelligence configuration plus reflection-phase feed refresh/recalibration into the structured personality-learning service without mixing that source logic into the core memory algorithms
- Split one service-level flush timeout into explicit per-writer budgets so wall-clock runtime deadlines stay real
- Keep low-signal multimodal writes on the deterministic reflection path while skipping optional midterm compilation that would otherwise duplicate per-turn remote latency
- Verify remote-primary snapshot readiness before runtime loops start
- Record per-snapshot pointer/origin readiness evidence so watchdog failures can be traced to the exact remote read path
- Reuse successful remote snapshot probes within one bounded readiness pass so watchdog startup does not refetch the same snapshot twice back-to-back
- Treat successful required snapshot loads as the decisive health proof inside the warm probe instead of re-running per-store backend status checks after bootstrap
- Split remote readiness into a strict bootstrap pass and a cheaper steady-state watchdog pass so live keepalives can prove current remote readability without reseeding every snapshot on every tick
- Skip the heavier `archive` snapshot during steady-state watchdog probes while keeping startup and recovery fully archive-inclusive and fail closed
- Fail closed when any required remote snapshot or shard is unreadable
- Prewarm generic foreground-read paths plus current object/conflict payload caches before live text-channel traffic so the first real remote-only recall turn can hit a warmed remote cache instead of rebuilding selectors and fetching first-hit payloads on demand
- Reuse remote snapshot/catalog/item reads through TTL-bounded in-process caches so warmed foreground recall stays sub-second on the Pi without changing remote-only truth semantics
- Keep foreground provider-context reads on the stores' own consistency boundaries instead of queueing them behind the shared background-writer mutation lock
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
| `service.py` | Runtime orchestration service |
| `worker.py` | Bounded async persistence |
| `flush_budget.py` | Total-deadline planner for multi-writer flush budgeting |
| `health.py` | Remote readiness probe with per-snapshot pointer/origin evidence |
| `component.yaml` | Structured package metadata |
| `AGENTS.md` | Local editing rules |

## Usage

```python
from twinr.memory.longterm import LongTermMemoryService

service = LongTermMemoryService.from_config(config)
service.ensure_remote_ready()
steady_state = service.probe_remote_ready(bootstrap=False, include_archive=False)
context = service.build_provider_context(query_text)
service.enqueue_conversation_turn(
    transcript=user_text,
    response=assistant_text,
)
service.record_personality_tool_history(
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
