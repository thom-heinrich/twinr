# core

Shared schemas and taxonomy helpers for Twinr long-term memory.

## Responsibility

`core` owns:
- Define long-term memory envelopes and dataclasses
- Normalize memory kinds and sensitivity labels

`core` does **not** own:
- Extract memories from turns or sensors
- Persist local or remote snapshots
- Orchestrate runtime workflows

## Key files

| File | Purpose |
|---|---|
| `__init__.py` | Mark the package |
| `models.py` | Define memory envelopes |
| `ontology.py` | Normalize kinds and labels |

## Usage

```python
from twinr.memory.longterm.core.models import LongTermMemoryObjectV1
from twinr.memory.longterm.core.ontology import normalize_memory_kind
```

## See also

- [../MEMORY_ARCHITECTURE.md](../MEMORY_ARCHITECTURE.md)
- [../runtime/README.md](../runtime/README.md)
