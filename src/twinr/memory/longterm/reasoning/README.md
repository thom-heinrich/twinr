# reasoning

Apply post-ingestion policy to long-term memory objects.

This package turns extracted candidates into durable state by handling
consolidation, truth maintenance, conflict resolution, reflection, midterm
compilation, immediate turn-continuity packets, and retention.

## Responsibility

`reasoning` owns:
- Consolidate extracted turn outputs into episodic, durable, deferred, and conflict results
- Maintain slot-level truth and build user-facing conflict choices
- Reflect over recent memory windows, including room-agnostic smart-home environment reflections, and compile bounded midterm packets
- Compile deterministic immediate turn-continuity packets from raw conversation turns so fresh follow-up recall does not wait on slower durable enrichment, while preserving short structured excerpts that downstream display/proactive code can reuse without parsing internal English packet text
- Apply retention, expiry, and archival policy to stored objects

Reflection must treat canonical proposition payloads as first-class evidence:
person-thread summaries cannot depend solely on hand-seeded `person_ref`,
`person_name`, or `fact_type=relationship` attributes when the raw turn
pipeline only grounded a person via `subject_ref`, `object_ref`, and
`predicate`.

When the ingestion layer has already compiled `smart_home_environment` day
profiles, baselines, grouped deviations, quality states, change points, regime
shifts, and node summaries, reflection also produces a bounded ambient-behavior
summary for the latest environment/day without assuming stable room labels.

Midterm packets keep canonical-English summaries/details for stable internal
reasoning, but their normalized `query_hints` must also preserve source-memory
phrases so first-turn retrieval does not depend on asynchronous query rewrites.
Immediate turn-continuity packets additionally preserve short structured raw
turn excerpts in `attributes` so display surfaces can build user-facing copy
without leaking those internal summary/details strings.
Retention also needs to treat raw multimodal `pattern` seeds differently from
day-scoped events: their `valid_to` marks the last observed day, not a next-day
expiry boundary.

`reasoning` does **not** own:
- Extract raw turns, propositions, or multimodal evidence
- Persist object, conflict, or archive snapshots
- Assemble retrieval context or proactive plans
- Run the top-level long-term runtime loop

## Key files

| File | Purpose |
|---|---|
| `consolidator.py` | Turn-level consolidation |
| `truth.py` | Slot conflict detection |
| `conflicts.py` | User conflict resolution |
| `reflect.py` | Reflection orchestration for person-thread and smart-home environment summaries |
| `midterm.py` | Midterm compiler adapter |
| `turn_continuity.py` | Deterministic immediate midterm packets from raw conversation turns |
| `retention.py` | Retention classification |
| `component.yaml` | Structural metadata |

## Usage

```python
from twinr.memory.longterm import (
    LongTermConflictResolver,
    LongTermMemoryConsolidator,
    LongTermMemoryReflector,
    LongTermRetentionPolicy,
    LongTermTruthMaintainer,
)

truth = LongTermTruthMaintainer()
consolidator = LongTermMemoryConsolidator(truth_maintainer=truth)
resolver = LongTermConflictResolver()
reflector = LongTermMemoryReflector.from_config(config)
retention = LongTermRetentionPolicy(timezone_name=config.local_timezone_name)
```

## See also

- [../ingestion/README.md](../ingestion/README.md)
- [../runtime/README.md](../runtime/README.md)
- [../storage/README.md](../storage/README.md)
- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
