# on_device

`on_device` owns Twinr's bounded short-term memory package. It defines the
typed records and in-memory store used for recent conversation context, compact
memory summaries, verified search results, and derived runtime state.

## Responsibility

`on_device` owns:
- define the canonical on-device memory records for turns, ledger items, search results, and state
- compact overflow conversation history into bounded summary-ready ledger entries
- sanitize legacy and structured snapshot payloads before rebuilding runtime memory
- expose defensive copies of memory data to callers

`on_device` does **not** own:
- snapshot file persistence or file loading
- long-term memory, graph memory, or remote-memory policy
- web rendering or high-level runtime orchestration

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Records and memory store |
| [component.yaml](./component.yaml) | Structured package metadata |

## Usage

```python
from twinr.memory import OnDeviceMemory

memory = OnDeviceMemory(max_turns=6, keep_recent=3)
memory.remember("user", "Wann faehrt der Bus?")
memory.remember_search(
    question="Wann faehrt der Bus nach Hamburg?",
    answer="Der Bus faehrt um 07:30 Uhr.",
    sources=("https://example.com/fahrplan",),
)
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [chonkydb](../chonkydb/README.md)
- [runtime snapshot mixin](../../agent/base_agent/runtime/snapshot.py)
