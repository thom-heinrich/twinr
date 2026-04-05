# evaluation

Isolated long-term memory evaluations for recall, multimodal retrieval, messy mixed-corpus recall/restart stress, unified retrieval quality, unified retrieval benchmarking, subtext behavior, live midterm memory attestation, live synthetic-memory acceptance, deploy-time live retention attestation, and forensic latency profiling of the remote long-term retrieval path.

## Responsibility

`evaluation` owns:
- deterministic evaluation fixture seeding and case definitions
- execution of synthetic, multimodal, unified-retrieval, and live subtext evals
- execution of live midterm write/read/usage attestations against the real OpenAI and ChonkyDB path in an isolated namespace, with remote proof taken from the authoritative midterm current head instead of a legacy snapshot blob
- execution of live synthetic-memory acceptance matrices covering earlier memory, conflict resolution, restart persistence, and control-query containment against the real OpenAI and ChonkyDB path in an isolated namespace
- seed live synthetic-memory acceptance fixtures through active-delta current-head writes instead of whole-state `write_snapshot(...)` rewrites, so the acceptance harness exercises the same bounded remote-write contract as the runtime paths it is attesting
- synchronously materialize the strict live provider-answer front for subtext cases before the answer turn, so the eval measures the real live-provider contract instead of failing on a missing transcript-first prewarm side effect
- keep live synthetic-memory acceptance focused on the low-latency memory surface by composing fast-topic hints, tool-safe durable context without graph fallback, and explicit conflict-queue rendering, so the proof does not trigger unrelated graph/world-state full-document reads while attesting memory recall
- execution of profiled unified-retrieval goldset cases that assert selected ids, join anchors, rendered sections, and access-path classes across durable, conflict, midterm, episodic, adaptive, and graph sources, with a narrow `core` suite for live acceptance and an `expanded` KPI suite with 50 natural-language recall cases
- execution of one large mixed-corpus remote-memory evaluation that composes the synthetic recall fixture, multimodal routine fixture, and expanded unified-retrieval goldset inside one shared ChonkyDB namespace, then reruns the same corpus from a fresh reader to expose restart regressions under messy cross-source distractors
- seed the large messy synthetic graph fixture locally and publish the finished graph once through the real remote graph persist path, so the benchmark measures retrieval quality under one shared corpus instead of spending most of its budget replaying 150 incremental full-graph rewrites
- execution of live writer/fresh-reader unified-retrieval acceptance against the real ChonkyDB path using the narrow `core` case profile by default
- execution of a unified-retrieval benchmark over the broader `expanded` case profile by default, reporting source-wise precision/recall, selected-id precision/recall, join-anchor quality, and path-safety metrics
- execution of a bounded live retention canary that seeds a fresh namespace, forbids broad object/snapshot hydration on the writer path, runs the real `run_retention()` flow, and proves the resulting current/archive heads from a fresh reader rooted at a separate runtime directory
- persist per-stage retention-canary evidence plus the latest watchdog observation so operators can tell whether a failing canary is contradicting the watchdog or simply proving a stricter isolated-namespace write/retention/readback contract
- execution of latency profiling runs that capture forensic workflow evidence for remote long-term retrieval bottlenecks, including fast pre-answer topic-hint reads, while bootstrapping required remote snapshot heads before the timed iterations when a fresh namespace has not yet been provisioned
- isolate per-case writable runtime state inside each temporary subtext-eval workspace so graph/runtime locks cannot bleed onto the shared repo state during live evaluation
- persist richer per-case subtext diagnostics, including query-profile variants, section previews, and seeded-term presence across subtext/durable/episodic/graph context, so answer-quality regressions can be separated from retrieval/bootstrap failures
- summarize live subtext runs with explicit execution-failure vs judge-failure counts plus personalization-context/seed-grounding coverage, so a `0/x` score can be classified as backend/bootstrap breakage, retrieval miss, or real answer-use weakness instead of collapsing into one opaque number
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
| `messy_memory_eval.py` | Large mixed-corpus remote-memory evaluation with writer/fresh-reader comparison |
| `_unified_retrieval_shared.py` | Shared fixture, profiled case catalogs, and case-evaluation helpers for unified retrieval quality checks |
| `unified_retrieval_goldset.py` | Deterministic unified-retrieval goldset runner against an isolated namespace, defaulting to the expanded KPI profile |
| `live_unified_retrieval_acceptance.py` | Live writer/fresh-reader acceptance for unified retrieval against real ChonkyDB, defaulting to the narrow core profile |
| `unified_retrieval_benchmark.py` | Precision/recall benchmark over the expanded unified-retrieval KPI profile |
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
from twinr.memory.longterm.evaluation.messy_memory_eval import run_messy_memory_eval
from twinr.memory.longterm.evaluation.multimodal_eval import run_multimodal_longterm_eval
from twinr.memory.longterm.evaluation.subtext_eval import run_subtext_response_eval
from twinr.memory.longterm.evaluation.unified_retrieval_benchmark import run_unified_retrieval_benchmark
from twinr.memory.longterm.evaluation.unified_retrieval_goldset import run_unified_retrieval_goldset

synthetic = run_synthetic_longterm_eval()
multimodal = run_multimodal_longterm_eval()
messy = run_messy_memory_eval(env_path=".env")
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

The multimodal eval keeps its fixture day relative to the run date so the
conversation-turn half stays inside the default episodic retention horizon
instead of silently aging out over calendar time.

Standard ops command:

```bash
PYTHONPATH=src python3 -m twinr --env-file .env --long-term-memory-live-acceptance
```

Unified retrieval goldset command:

```bash
PYTHONPATH=src ./.venv/bin/python -m twinr.memory.longterm.evaluation.unified_retrieval_goldset --env-file .env --case-profile expanded
```

Unified retrieval live acceptance command:

```bash
PYTHONPATH=src ./.venv/bin/python -m twinr.memory.longterm.evaluation.live_unified_retrieval_acceptance --env-file .env --case-profile core
```

Unified retrieval benchmark command:

```bash
PYTHONPATH=src ./.venv/bin/python -m twinr.memory.longterm.evaluation.unified_retrieval_benchmark --env-file .env --case-profile expanded
```

Messy mixed-corpus evaluation command:

```bash
PYTHONPATH=src ./.venv/bin/python -m twinr.memory.longterm.evaluation.messy_memory_eval --env-file .env --unified-case-profile expanded
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
Pi deploy helper as a post-restart remote-memory proof. The report now records
`failure_stage`, per-step latencies, and the latest watchdog observation so a
green watchdog plus a red canary reads as “different proof surfaces” instead of
two contradictory truths.
When the remote canary itself takes longer than the caller's SSH stdout budget,
the deploy path can recover this per-run report by `probe_id` and keep the
operator-facing error aligned with the Pi-side structured result.

The unified benchmark persists the latest benchmark summary to
`artifacts/stores/ops/unified_retrieval_benchmark.json` and a per-run snapshot
under `artifacts/reports/unified_retrieval_benchmark/`. It is intended to make
over-selection and join-quality regressions visible even when the pass/fail
goldset still succeeds.

The messy mixed-corpus runner persists the latest result to
`artifacts/stores/ops/messy_memory_eval.json` and a per-run snapshot under
`artifacts/reports/messy_memory_eval/`. It is intended to answer a stricter
question than the individual suites: whether the same retrieval quality still
holds once synthetic graph clutter, multimodal routines, and unified
multi-source conflicts all coexist in one remote namespace and are then read
back by a fresh runtime. During the scored writer/fresh-reader phases it
deliberately disables optional query-rewrite and subtext-compiler LLM lanes and
does not persist materialized provider-answer fronts, so the benchmark isolates
remote memory quality instead of auxiliary cache/LLM latency.

The unified-retrieval profiles expose two operational tiers:
- `core`: 3 high-signal cases for bounded live writer/fresh-reader acceptance.
- `expanded`: 50 natural-language recall cases with per-memory-type coverage of at least 30 cases for adaptive, conflict, durable, episodic, graph, and midterm sources.

The latency profiler persists a condensed summary to
`artifacts/reports/longterm_latency_profile/<profile_id>/profile.json` and the
correlated workflow run pack under the sibling `workflow/` directory.

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [../MEMORY_ARCHITECTURE.md](../MEMORY_ARCHITECTURE.md)
- [../runtime/README.md](../runtime/README.md)
