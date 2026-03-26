# runtime

Runtime-facing orchestration for Twinr long-term memory.

This package wires stores, reasoning helpers, remote readiness checks, and
bounded background writers into the APIs used by agent runtime loops.

## Responsibility

`runtime` owns:
- Assemble `LongTermMemoryService` from config and subsystem dependencies
- Build runtime provider context and tool-safe context while preserving both original-language and canonical query recall paths
- Inject the bounded fast-topic quick-memory lane into the normal provider context so ordinary answer turns receive a few compact personalized topic hints before the heavier recall sections
- Build one bounded fast-topic provider context for direct/latency-sensitive reply lanes that only need a few current-topic hints before answering, while surfacing specific required remote-read failures separately from broader backend-unavailable state
- Expose confirmed durable-memory state explicitly in provider/tool context so meta-memory questions can distinguish stored current facts from generic neighbors
- Rebuild and persist provenance-rich restart-recall packets after durable-memory mutations so fresh runtime roots retain immediate continuity
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
