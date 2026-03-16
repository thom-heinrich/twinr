# proactive

Plan and persist long-term proactive suggestions and reservations.

## Responsibility

`proactive` owns:
- Rank proactive candidates from memory state
- Persist proactive history and reservations

`proactive` does **not** own:
- Extract memories from raw inputs
- Store durable memory objects
- Run the outer runtime orchestration loop

## Key files

| File | Purpose |
|---|---|
| `__init__.py` | Preserve package exports |
| `planner.py` | Rank proactive candidates |
| `state.py` | Persist proactive state |

## Usage

```python
from twinr.memory.longterm.proactive.planner import LongTermProactivePlanner
from twinr.memory.longterm.proactive.state import LongTermProactivePolicy
```

## See also

- [../MEMORY_ARCHITECTURE.md](../MEMORY_ARCHITECTURE.md)
- [../runtime/README.md](../runtime/README.md)
