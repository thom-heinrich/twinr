# core

Canonical schemas and taxonomy helpers for Twinr long-term memory.

## Responsibility

`core` owns:
- versioned long-term memory dataclasses and schema constants
- kind, domain, and sensitivity normalization helpers
- payload validation and serialization for shared memory contracts

`core` does **not** own:
- extraction, consolidation, or retrieval workflows
- persistence engines or background writers
- evaluation harnesses or benchmark fixtures

## Key files

| File | Purpose |
|---|---|
| `models.py` | Versioned memory schemas |
| `ontology.py` | Taxonomy normalization |
| `component.yaml` | Structured ownership metadata |
| `AGENTS.md` | Local editing rules |

## Usage

```python
from twinr.memory.longterm.core.models import LongTermMemoryObjectV1, LongTermSourceRefV1
from twinr.memory.longterm.core.ontology import normalize_memory_kind

source = LongTermSourceRefV1(source_type="conversation", event_ids=("turn-1",))
memory = LongTermMemoryObjectV1(
    memory_id="mem-1",
    kind="relationship_fact",
    summary="Corinna is the physiotherapist.",
    source=source,
)
kind, attributes = normalize_memory_kind(memory.kind, memory.attributes)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [../MEMORY_ARCHITECTURE.md](../MEMORY_ARCHITECTURE.md)
- [../runtime/README.md](../runtime/README.md)
