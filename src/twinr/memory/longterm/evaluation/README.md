# evaluation

Isolated long-term memory evaluations for recall, multimodal retrieval, and subtext behavior.

## Responsibility

`evaluation` owns:
- deterministic evaluation fixture seeding and case definitions
- execution of synthetic, multimodal, and live subtext evals
- structured result payloads and persisted artifact snapshots

`evaluation` does **not** own:
- runtime memory serving or production-state writes
- core schema or taxonomy contracts
- top-level test assertions outside `test/`

## Key files

| File | Purpose |
|---|---|
| `eval.py` | Synthetic recall eval |
| `multimodal_eval.py` | Multimodal retrieval eval |
| `subtext_eval.py` | Live subtext eval |
| `component.yaml` | Structured ownership metadata |

## Usage

```python
from twinr.memory.longterm.evaluation.eval import run_synthetic_longterm_eval
from twinr.memory.longterm.evaluation.multimodal_eval import run_multimodal_longterm_eval
from twinr.memory.longterm.evaluation.subtext_eval import run_subtext_response_eval

synthetic = run_synthetic_longterm_eval()
multimodal = run_multimodal_longterm_eval()
subtext = run_subtext_response_eval(env_path=".env")
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [../MEMORY_ARCHITECTURE.md](../MEMORY_ARCHITECTURE.md)
- [../runtime/README.md](../runtime/README.md)
