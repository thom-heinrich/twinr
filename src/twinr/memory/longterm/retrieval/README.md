# retrieval

Build recall context and silent personalization cues from long-term memory state.

## Responsibility

`retrieval` owns:
- Retrieve durable, episodic, and midterm context
- Compile subtext and personalization payloads

`retrieval` does **not** own:
- Persist memory state
- Extract new candidate memories
- Run proactive planning or runtime orchestration

## Key files

| File | Purpose |
|---|---|
| `retriever.py` | Build prompt context |
| `subtext.py` | Compile subtext cues |

## Usage

```python
from twinr.memory.longterm.retrieval.retriever import LongTermRetriever
from twinr.memory.longterm.retrieval.subtext import LongTermSubtextBuilder
```

## See also

- [../MEMORY_ARCHITECTURE.md](../MEMORY_ARCHITECTURE.md)
- [../storage/README.md](../storage/README.md)
