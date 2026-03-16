# reasoning

Apply consolidation, truth, retention, reflection, and conflict logic to long-term memories.

## Responsibility

`reasoning` owns:
- Consolidate candidate memories into durable objects
- Resolve conflicts and maintain truth state
- Reflect, retain, and archive memory state

`reasoning` does **not** own:
- Capture raw turn or sensor evidence
- Store snapshots or archives directly
- Run the top-level long-term service loop

## Key files

| File | Purpose |
|---|---|
| `conflicts.py` | Resolve contradictions |
| `consolidator.py` | Promote candidate memories |
| `midterm.py` | Build reflection prompts |
| `reflect.py` | Generate reflections |
| `retention.py` | Apply retention policy |
| `truth.py` | Maintain truth state |

## Usage

```python
from twinr.memory.longterm.reasoning.consolidator import LongTermMemoryConsolidator
from twinr.memory.longterm.reasoning.truth import LongTermTruthMaintainer
```

## See also

- [../MEMORY_ARCHITECTURE.md](../MEMORY_ARCHITECTURE.md)
- [../ingestion/README.md](../ingestion/README.md)
