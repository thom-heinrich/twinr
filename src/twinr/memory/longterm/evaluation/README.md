# evaluation

Isolated long-term memory evaluations for recall, multimodal retrieval, unified retrieval quality, unified retrieval benchmarking, subtext behavior, live midterm memory attestation, live synthetic-memory acceptance, deploy-time live retention attestation, and forensic latency profiling of the remote long-term retrieval path.

## Responsibility

`evaluation` owns:
- deterministic evaluation fixture seeding and case definitions
- execution of synthetic, multimodal, unified-retrieval, and live subtext evals
- execution of live midterm write/read/usage attestations against the real OpenAI and ChonkyDB path in an isolated namespace, with remote proof taken from the authoritative midterm current head instead of a legacy snapshot blob
- execution of live synthetic-memory acceptance matrices covering earlier memory, conflict resolution, restart persistence, and control-query containment against the real OpenAI and ChonkyDB path in an isolated namespace
- seed live synthetic-memory acceptance fixtures through active-delta current-head writes instead of whole-state `write_snapshot(...)` rewrites, so the acceptance harness exercises the same bounded remote-write contract as the runtime paths it is attesting
- execution of fixed unified-retrieval goldset cases that assert selected ids, join anchors, rendered sections, and access-path classes across durable, conflict, midterm, episodic, adaptive, and graph sources
- execution of live writer/fresh-reader unified-retrieval acceptance against the real ChonkyDB path using the same fixed goldset cases as the local goldset runner
- execution of a unified-retrieval benchmark over the same fixed goldset cases, reporting source-wise precision/recall, selected-id precision/recall, join-anchor quality, and path-safety metrics
- execution of a bounded live retention canary that seeds a fresh namespace, forbids broad object/snapshot hydration on the writer path, runs the real `run_retention()` flow, and proves the resulting current/archive heads from a fresh reader rooted at a separate runtime directory
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
| `_unified_retrieval_shared.py` | Shared fixed fixture, cases, and case-evaluation helpers for unified retrieval quality checks |
| `unified_retrieval_goldset.py` | Deterministic unified-retrieval goldset runner against an isolated namespace |
| `live_unified_retrieval_acceptance.py` | Live writer/fresh-reader acceptance for unified retrieval against real ChonkyDB |
| `unified_retrieval_benchmark.py` | Precision/recall benchmark over the fixed unified-retrieval goldset |
| `subtext_eval.py` | Live subtext eval |
| `live_midterm_acceptance.py` | Live midterm memory E2E attestation runner |
| `live_midterm_attest.py` | Artifact/result contract for the live midterm attestation |
| `live_memory_acceptance.py` | Live synthetic-memory acceptance runner for earlier/conflict/restart/control cases |
| `live_retention_canary.py` | Bounded live retention canary for deploy-time remote retention attestation |
| `latency_profile.py` | Forensic latency profiler for provider-context and fast-provider-context retrieval with a required-remote readiness/bootstrap preflight outside the measured iterations |
| `component.yaml` | Structured ownership metadata |

## Usage

```python
from twinr.memory.longterm.evaluation.eval import run_synthetic_longterm_eval
from twinr.memory.longterm.evaluation.latency_profile import run_longterm_latency_profile
from twinr.memory.longterm.evaluation.live_memory_acceptance import run_live_memory_acceptance
from twinr.memory.longterm.evaluation.live_midterm_acceptance import run_live_midterm_acceptance
from twinr.memory.longterm.evaluation.live_unified_retrieval_acceptance import run_live_unified_retrieval_acceptance
from twinr.memory.longterm.evaluation.live_retention_canary import run_live_retention_canary
from twinr.memory.longterm.evaluation.multimodal_eval import run_multimodal_longterm_eval
from twinr.memory.longterm.evaluation.subtext_eval import run_subtext_response_eval
from twinr.memory.longterm.evaluation.unified_retrieval_benchmark import run_unified_retrieval_benchmark
from twinr.memory.longterm.evaluation.unified_retrieval_goldset import run_unified_retrieval_goldset

synthetic = run_synthetic_longterm_eval()
multimodal = run_multimodal_longterm_eval()
goldset = run_unified_retrieval_goldset(env_path=".env")
benchmark = run_unified_retrieval_benchmark(env_path=".env")
subtext = run_subtext_response_eval(env_path=".env")
attest = run_live_midterm_acceptance(env_path=".env")
acceptance = run_live_memory_acceptance(env_path=".env")
unified_acceptance = run_live_unified_retrieval_acceptance(env_path=".env")
retention = run_live_retention_canary(env_path=".env")
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

Unified retrieval goldset command:

```bash
PYTHONPATH=src ./.venv/bin/python -m twinr.memory.longterm.evaluation.unified_retrieval_goldset --env-file .env
```

Unified retrieval live acceptance command:

```bash
PYTHONPATH=src ./.venv/bin/python -m twinr.memory.longterm.evaluation.live_unified_retrieval_acceptance --env-file .env
```

Unified retrieval benchmark command:

```bash
PYTHONPATH=src ./.venv/bin/python -m twinr.memory.longterm.evaluation.unified_retrieval_benchmark --env-file .env
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

The retention canary persists a bounded per-run snapshot under
`artifacts/reports/retention_live_canary/` and is intended to be called by the
Pi deploy helper as a post-restart remote-memory proof.

The unified benchmark persists the latest benchmark summary to
`artifacts/stores/ops/unified_retrieval_benchmark.json` and a per-run snapshot
under `artifacts/reports/unified_retrieval_benchmark/`. It is intended to make
over-selection and join-quality regressions visible even when the pass/fail
goldset still succeeds.

The latency profiler persists a condensed summary to
`artifacts/reports/longterm_latency_profile/<profile_id>/profile.json` and the
correlated workflow run pack under the sibling `workflow/` directory.

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [../MEMORY_ARCHITECTURE.md](../MEMORY_ARCHITECTURE.md)
- [../runtime/README.md](../runtime/README.md)
