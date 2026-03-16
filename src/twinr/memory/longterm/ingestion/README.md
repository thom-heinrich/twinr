# ingestion

Extract candidate long-term memories from conversations, sensors, and multimodal evidence.

## Responsibility

`ingestion` owns:
- Parse turns into structured propositions
- Compile sensor and multimodal evidence
- Backfill ops events into memory inputs

`ingestion` does **not** own:
- Persist durable objects or archives
- Resolve conflicts or truth state
- Build retrieval context for prompts

## Key files

| File | Purpose |
|---|---|
| `backfill.py` | Replay ops events |
| `extract.py` | Extract turn memories |
| `multimodal.py` | Compile multimodal evidence |
| `propositions.py` | Build proposition bundles |
| `sensor_memory.py` | Compile sensor routines |

## Usage

```python
from twinr.memory.longterm.ingestion.extract import LongTermTurnExtractor
from twinr.memory.longterm.ingestion.multimodal import LongTermMultimodalExtractor
```

## See also

- [../MEMORY_ARCHITECTURE.md](../MEMORY_ARCHITECTURE.md)
- [../reasoning/README.md](../reasoning/README.md)
