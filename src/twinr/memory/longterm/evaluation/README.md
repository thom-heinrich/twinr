# evaluation

Run targeted long-term memory evaluations without changing runtime code paths.

## Responsibility

`evaluation` owns:
- Seed synthetic long-term memory scenarios
- Score retrieval and subtext behavior
- Snapshot evaluation outputs for inspection

`evaluation` does **not** own:
- Serve runtime memory requests
- Persist production memory state
- Normalize core schemas or taxonomy

## Key files

| File | Purpose |
|---|---|
| `eval.py` | Run synthetic recall evals |
| `multimodal_eval.py` | Run multimodal evals |
| `subtext_eval.py` | Judge subtext responses |

## Usage

```python
from twinr.memory.longterm.evaluation.eval import run_synthetic_longterm_eval
from twinr.memory.longterm.evaluation.subtext_eval import run_subtext_response_eval
```

## See also

- [../MEMORY_ARCHITECTURE.md](../MEMORY_ARCHITECTURE.md)
- [../runtime/README.md](../runtime/README.md)
