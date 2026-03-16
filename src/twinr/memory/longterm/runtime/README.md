# runtime

Wire the long-term memory subsystem into Twinr runtime flows and bounded background workers.

## Responsibility

`runtime` owns:
- Orchestrate long-term memory dependencies
- Expose the main long-term memory service
- Buffer async writes in bounded workers

`runtime` does **not** own:
- Define memory schemas or taxonomy
- Implement storage backends directly
- Hold evaluation harnesses

## Key files

| File | Purpose |
|---|---|
| `service.py` | Orchestrate long-term flows |
| `worker.py` | Run bounded async writers |

## Usage

```python
from twinr.memory.longterm.runtime.service import LongTermMemoryService
from twinr.memory.longterm.runtime.worker import AsyncLongTermMemoryWriter
```

## See also

- [../MEMORY_ARCHITECTURE.md](../MEMORY_ARCHITECTURE.md)
- [../core/README.md](../core/README.md)
