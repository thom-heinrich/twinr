# evaluation

Isolated long-term memory evaluations for recall, multimodal retrieval, subtext behavior, live midterm memory attestation, and live synthetic-memory acceptance.

## Responsibility

`evaluation` owns:
- deterministic evaluation fixture seeding and case definitions
- execution of synthetic, multimodal, and live subtext evals
- execution of live midterm write/read/usage attestations against the real OpenAI and ChonkyDB path in an isolated namespace
- execution of live synthetic-memory acceptance matrices covering earlier memory, conflict resolution, restart persistence, and control-query containment against the real OpenAI and ChonkyDB path in an isolated namespace
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
| `live_midterm_acceptance.py` | Live midterm memory E2E attestation runner |
| `live_midterm_attest.py` | Artifact/result contract for the live midterm attestation |
| `live_memory_acceptance.py` | Live synthetic-memory acceptance runner for earlier/conflict/restart/control cases |
| `component.yaml` | Structured ownership metadata |

## Usage

```python
from twinr.memory.longterm.evaluation.eval import run_synthetic_longterm_eval
from twinr.memory.longterm.evaluation.live_memory_acceptance import run_live_memory_acceptance
from twinr.memory.longterm.evaluation.live_midterm_acceptance import run_live_midterm_acceptance
from twinr.memory.longterm.evaluation.multimodal_eval import run_multimodal_longterm_eval
from twinr.memory.longterm.evaluation.subtext_eval import run_subtext_response_eval

synthetic = run_synthetic_longterm_eval()
multimodal = run_multimodal_longterm_eval()
subtext = run_subtext_response_eval(env_path=".env")
attest = run_live_midterm_acceptance(env_path=".env")
acceptance = run_live_memory_acceptance(env_path=".env")
```

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [../MEMORY_ARCHITECTURE.md](../MEMORY_ARCHITECTURE.md)
- [../runtime/README.md](../runtime/README.md)
