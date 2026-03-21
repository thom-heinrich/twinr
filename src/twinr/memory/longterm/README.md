# longterm

Package root for Twinr's long-term memory subsystem. It exposes the canonical
`twinr.memory.longterm` import surface and points newcomers to the subpackages
that own models, ingestion, reasoning, retrieval, storage, runtime, proactive
behavior, and evaluation. ReSpeaker-specific audio routine learning is ingested
through structured multimodal facts under `ingestion/`; no raw PCM belongs in
this subsystem. ReSpeaker-derived observed preferences remain distinct from
explicitly confirmed preferences and must not auto-promote just because their
confidence is high. Confirmed response-channel preferences can then flow back
out through `retrieval/` as explicit adaptive policy packets. Room-agnostic
smart-home environment profiling also compiles under `ingestion/`, using
motion-node, transition, fragmentation, and timing markers instead of hard
room labels.

## Responsibility

`longterm` owns:
- expose the supported package-level import surface
- define the top-level boundaries between the long-term memory subpackages
- hold the subsystem architecture overview

`longterm` does **not** own:
- short-term/on-device memory or prompt-context stores
- low-level ChonkyDB client and graph-schema code outside long-term storage integration
- top-level agent loops, web routes, or hardware orchestration

## Key files

| File | Purpose |
|---|---|
| `__init__.py` | Package exports and lazy eval bridge |
| `MEMORY_ARCHITECTURE.md` | Target-state long-term design |
| `core/` | Models and ontology helpers |
| `ingestion/` | Turn and multimodal extraction |
| `reasoning/` | Truth, reflection, retention, conflicts |
| `retrieval/` | Retrieval and subtext assembly |
| `storage/` | Stores and remote snapshots |
| `runtime/` | Runtime service and readiness |
| `proactive/` | Proactive planning and state |
| `evaluation/` | Synthetic and multimodal evals |

## Usage

```python
from twinr.memory.longterm import LongTermMemoryService

service = LongTermMemoryService.from_config(config)
service.ensure_remote_ready()
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [MEMORY_ARCHITECTURE.md](./MEMORY_ARCHITECTURE.md)
- [runtime/README.md](./runtime/README.md)
- [storage/README.md](./storage/README.md)
