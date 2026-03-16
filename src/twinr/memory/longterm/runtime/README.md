# runtime

Runtime-facing orchestration for Twinr long-term memory.

This package wires stores, reasoning helpers, remote readiness checks, and
bounded background writers into the APIs used by agent runtime loops.

## Responsibility

`runtime` owns:
- Assemble `LongTermMemoryService` from config and subsystem dependencies
- Build runtime provider context and tool-safe context
- Persist conversation turns and multimodal evidence through bounded workers
- Verify remote-primary snapshot readiness before runtime loops start
- Fail closed when any required remote snapshot or shard is unreadable
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
| `health.py` | Remote readiness probe |
| `component.yaml` | Structured package metadata |
| `AGENTS.md` | Local editing rules |

## Usage

```python
from twinr.memory.longterm import LongTermMemoryService

service = LongTermMemoryService.from_config(config)
service.ensure_remote_ready()
context = service.build_provider_context(query_text)
service.enqueue_conversation_turn(
    transcript=user_text,
    response=assistant_text,
)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [../MEMORY_ARCHITECTURE.md](../MEMORY_ARCHITECTURE.md)
- [../ingestion/README.md](../ingestion/README.md)
- [../reasoning/README.md](../reasoning/README.md)
- [../storage/README.md](../storage/README.md)
