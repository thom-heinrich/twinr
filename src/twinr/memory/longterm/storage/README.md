# storage

Persist long-term memory objects, midterm packets, and remote snapshots.

## Responsibility

`storage` owns:
- Store durable objects and archives
- Read and write midterm packets
- Bridge to remote ChonkyDB-backed state

`storage` does **not** own:
- Extract or normalize new memories
- Decide truth, retention, or conflicts
- Orchestrate runtime memory flows

## Key files

| File | Purpose |
|---|---|
| `midterm_store.py` | Persist midterm packets |
| `remote_state.py` | Read remote snapshots |
| `store.py` | Persist durable objects |

## Usage

```python
from twinr.memory.longterm.storage.store import LongTermStructuredStore
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore
```

## See also

- [../MEMORY_ARCHITECTURE.md](../MEMORY_ARCHITECTURE.md)
- [../runtime/README.md](../runtime/README.md)
