# evaluation

Isolated long-term memory evaluations for recall, multimodal retrieval, subtext behavior, live midterm memory attestation, live synthetic-memory acceptance, and forensic latency profiling of the remote long-term retrieval path.

## Responsibility

`evaluation` owns:
- deterministic evaluation fixture seeding and case definitions
- execution of synthetic, multimodal, and live subtext evals
- execution of live midterm write/read/usage attestations against the real OpenAI and ChonkyDB path in an isolated namespace, with remote proof taken from the authoritative midterm current head instead of a legacy snapshot blob
- execution of live synthetic-memory acceptance matrices covering earlier memory, conflict resolution, restart persistence, and control-query containment against the real OpenAI and ChonkyDB path in an isolated namespace
- execution of latency profiling runs that capture forensic workflow evidence for remote long-term retrieval bottlenecks, including fast pre-answer topic-hint reads, while bootstrapping required remote snapshot heads before the timed iterations when a fresh namespace has not yet been provisioned
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
| `latency_profile.py` | Forensic latency profiler for provider-context and fast-provider-context retrieval with a required-remote readiness/bootstrap preflight outside the measured iterations |
| `component.yaml` | Structured ownership metadata |

## Usage

```python
from twinr.memory.longterm.evaluation.eval import run_synthetic_longterm_eval
from twinr.memory.longterm.evaluation.latency_profile import run_longterm_latency_profile
from twinr.memory.longterm.evaluation.live_memory_acceptance import run_live_memory_acceptance
from twinr.memory.longterm.evaluation.live_midterm_acceptance import run_live_midterm_acceptance
from twinr.memory.longterm.evaluation.multimodal_eval import run_multimodal_longterm_eval
from twinr.memory.longterm.evaluation.subtext_eval import run_subtext_response_eval

synthetic = run_synthetic_longterm_eval()
multimodal = run_multimodal_longterm_eval()
subtext = run_subtext_response_eval(env_path=".env")
attest = run_live_midterm_acceptance(env_path=".env")
acceptance = run_live_memory_acceptance(env_path=".env")
profile = run_longterm_latency_profile(
    env_path=".env",
    queries=["Was weisst du ueber xyz?"],
    runs_per_query=3,
    target="fast_provider_context",
)
```

Standard ops command:

```bash
PYTHONPATH=src python3 -m twinr --env-file .env --long-term-memory-live-acceptance
```

Latency profiling command:

```bash
PYTHONPATH=src python3 -m twinr.memory.longterm.evaluation.latency_profile \
  --env-file .env \
  --query "Was weisst du ueber xyz?" \
  --target fast_provider_context \
  --runs 3
```

The standard command persists the latest result to
`artifacts/stores/ops/memory_live_acceptance.json` and a per-run snapshot under
`artifacts/reports/memory_live_acceptance/`.

The latency profiler persists a condensed summary to
`artifacts/reports/longterm_latency_profile/<profile_id>/profile.json` and the
correlated workflow run pack under the sibling `workflow/` directory.

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [../MEMORY_ARCHITECTURE.md](../MEMORY_ARCHITECTURE.md)
- [../runtime/README.md](../runtime/README.md)
