# ingestion

Normalize conversation turns, multimodal evidence, and replayed ops history into
candidate long-term memory inputs.

## Responsibility

`ingestion` owns:
- Extract an episode plus candidate objects and graph edges from one turn
- Normalize multimodal sensor, button, printer, and camera evidence
- Replay persisted ops history into multimodal evidence for backfill
- Compile sensor-derived routines and deviations from pattern history

`ingestion` does **not** own:
- Persist long-term objects, archives, or remote state
- Resolve conflicts, truth state, retention, or reflection policy
- Orchestrate runtime queues, workers, or service lifecycles
- Build retrieval context for downstream prompts

## Key files

| File | Purpose |
|---|---|
| `extract.py` | Turn extraction entrypoint |
| `propositions.py` | Structured proposition compiler |
| `multimodal.py` | Multimodal event extractor |
| `sensor_memory.py` | Sensor routine compiler |
| `backfill.py` | Ops history replay |

## Usage

```python
from twinr.memory.longterm.ingestion.backfill import LongTermOpsEventBackfiller
from twinr.memory.longterm.ingestion.extract import LongTermTurnExtractor
from twinr.memory.longterm.ingestion.multimodal import LongTermMultimodalExtractor
from twinr.memory.longterm.ingestion.sensor_memory import LongTermSensorMemoryCompiler
```

## See also

- [../MEMORY_ARCHITECTURE.md](../MEMORY_ARCHITECTURE.md)
- [../runtime/README.md](../runtime/README.md)
- [../reasoning/README.md](../reasoning/README.md)
- [AGENTS.md](./AGENTS.md)
- [component.yaml](./component.yaml)
