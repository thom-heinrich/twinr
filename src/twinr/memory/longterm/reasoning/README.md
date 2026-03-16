# reasoning

Apply post-ingestion policy to long-term memory objects.

This package turns extracted candidates into durable state by handling
consolidation, truth maintenance, conflict resolution, reflection, midterm
compilation, and retention.

## Responsibility

`reasoning` owns:
- Consolidate extracted turn outputs into episodic, durable, deferred, and conflict results
- Maintain slot-level truth and build user-facing conflict choices
- Reflect over recent memory windows and compile bounded midterm packets
- Apply retention, expiry, and archival policy to stored objects

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
| `reflect.py` | Reflection orchestration |
| `midterm.py` | Midterm compiler adapter |
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
