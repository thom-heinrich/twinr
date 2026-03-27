# self_coding

`self_coding` owns the focused package slice of Twinr's Adaptive Skill Engine
(ASE). The current scope covers the deterministic front-stage core plus the
first local compile-worker, `skill_package` runtime, and activation slices:
versioned contracts, status enums, a file-backed runtime store, the MVP
capability registry, the curated MVP module library under `modules/`,
feasibility checker, the requirements-dialogue state machine, queued compile
jobs, bounded compile workspaces, deterministic `automation_manifest` and
`skill_package` compiler paths, compile status persistence, versioned
soft-launch activation plus pause/reactivate/rollback/cleanup control,
persisted skill health counters with bounded auto-pause, the first brokered
process-isolated sandbox for `skill_package`, explicit capability-policy
manifests per compiled skill, best-effort child-process hardening
(`no_new_privs` plus Landlock where available), persisted execution-run and
watchdog state, a compact operator-status summary, capture-only operator
retests, and the local Codex drivers used to draft artifacts. The primary
local driver is a pinned `codex-sdk` bridge using `gpt-5-codex` with `high`
reasoning effort by default for self-coding compiles, with `codex exec --json`
kept as the bounded fallback path.

## Responsibility

`self_coding` owns:
- the stable internal contract objects used across future ASE layers
- the status model for capabilities, compile jobs, artifacts, and learned skills
- the file-backed store rooted at `state/self_coding`
- the persisted compile-status record used for operator/debug visibility
- operator-facing activation, health, execution-run, compile-status, and live-e2e records that stay readable across root/runtime and Pi-web service boundaries
- the deterministic capability registry for the MVP module set
- the Codex-readable trusted module-library shims under `modules/`
- the deterministic feasibility checker for Phase 1
- the persistent requirements-dialogue flow for Phase 2
- the compile-worker that creates queued jobs and persists local Codex artifacts
- the compiler paths and validator that canonicalize `automation_manifest` and `skill_package` outputs
- the deterministic compile-prompt builder that anchors Codex on Twinr artifact contracts
- versioned soft-launch staging, confirmation, rollback, pause, and reactivation for learned skills
- the materialized `skill_package` runtime, trusted sandbox loader, broker, and bounded `ctx.*` execution surface
- the explicit broker-policy manifest that narrows the runtime `ctx.*` surface per required capability
- best-effort child-process hardening and persisted hardening reports for sandboxed skill execution
- persisted skill health counters, live-e2e proof records, and bounded auto-pause policy
- persisted execution-run records plus stale-run watchdog visibility and cleanup helpers
- the compact operator-status summary and telemetry consumed by the web dashboard and self-coding ops page
- operator-triggered rollback, retest, stale-run cleanup, and cleanup control paths for learned skill versions
- the bounded local Codex workspaces and driver adapters under `codex_driver/`
- the machine-readiness preflight for the pinned Codex SDK bridge and auth on local/Pi runtimes
- the live morning-briefing acceptance runner that proves compile, refresh, delivery, and spoken output

`self_coding` does **not** own:
- workflow-loop orchestration or runtime hooks
- richer health scoring and review workflow beyond the current bounded counter-based policy
- compile-target specialization beyond the current deterministic validator plus `automation_manifest`/`skill_package` slice
- deeper seccomp/filter-style sandbox hardening beyond the current best-effort `no_new_privs` plus Landlock slice

## Key files

| File | Purpose |
|---|---|
| [__init__.py](./__init__.py) | Package export surface |
| [contracts.py](./contracts.py) | Versioned ASE contract dataclasses |
| [status.py](./status.py) | Shared ASE enums and lifecycle states |
| [store.py](./store.py) | File-backed store under `state/self_coding`, including cross-service-readable operator status records for the Pi web portal |
| [compiler/](./compiler) | Deterministic compile-target validation and canonicalization |
| [compiler/prompting.py](./compiler/prompting.py) | Deterministic compile-prompt builder for Twinr artifact contracts |
| [capability_registry.py](./capability_registry.py) | Deterministic MVP capability readiness view |
| [modules/](./modules) | Curated Codex-readable module library for MVP ASE capabilities |
| [feasibility.py](./feasibility.py) | Deterministic Phase-1 feasibility checker |
| [requirements_dialogue.py](./requirements_dialogue.py) | Deterministic Phase-2 dialogue state machine |
| [learning_flow.py](./learning_flow.py) | Thin orchestration of feasibility, dialogue, and queued compile creation |
| [worker.py](./worker.py) | Queued compile-job execution, status updates, and artifact persistence |
| [activation.py](./activation.py) | Versioned soft-launch staging, enablement, rollback, pause, reactivation, and cleanup |
| [execution_runs.py](./execution_runs.py) | Persist sandbox/retest run lifecycle transitions |
| [health.py](./health.py) | Skill health counters, live-e2e proofs, and bounded auto-pause policy |
| [sandbox/](./sandbox) | AST validator, trusted loader, capability broker, and child-process runner |
| [runtime/](./runtime) | Materialized `skill_package` runtime and bounded execution context |
| [retest.py](./retest.py) | Capture-only operator retest runner for active learned skills |
| [operator_status.py](./operator_status.py) | Compact operator-facing compile, health, and live-e2e summary |
| [watchdog.py](./watchdog.py) | Stale compile/run visibility plus operator cleanup helpers |
| [codex_driver/](./codex_driver) | Pinned Codex SDK bridge, exec fallback, and workspace builders |
| [codex_driver/environment.py](./codex_driver/environment.py) | Codex SDK bridge, CLI, and auth readiness checks |
| [live_acceptance.py](./live_acceptance.py) | Real morning-briefing compile/run acceptance path |
| [component.yaml](./component.yaml) | Structured package metadata |
| [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) | Phased work packages |
| [REQUIREMENTS.md](./REQUIREMENTS.md) | Product and architecture requirements |

## Usage

```python
from twinr.agent.self_coding import (
    CompileJobRecord,
    CompileJobStatus,
    CompileRunStatusRecord,
    CompileTarget,
    LocalCodexCompileDriver,
    SelfCodingActivationService,
    SelfCodingCompileWorker,
    SelfCodingFeasibilityChecker,
    SelfCodingLearningFlow,
    build_self_coding_operator_status,
    SelfCodingCapabilityRegistry,
    SelfCodingRequirementsDialogue,
    SelfCodingStore,
)

store = SelfCodingStore.from_project_root(".")
registry = SelfCodingCapabilityRegistry(project_root=".")
checker = SelfCodingFeasibilityChecker(registry)
dialogue = SelfCodingRequirementsDialogue()
worker = SelfCodingCompileWorker(store=store)
flow = SelfCodingLearningFlow(store=store, checker=checker, dialogue=dialogue, compile_worker=worker)
driver = LocalCodexCompileDriver()
status = build_self_coding_operator_status(store)
job = CompileJobRecord(
    job_id="job_demo123",
    skill_id="read_messages",
    skill_name="Read Messages",
    status=CompileJobStatus.QUEUED,
    requested_target=CompileTarget.AUTOMATION_MANIFEST,
    spec_hash="0" * 64,
)
store.save_job(job)
store.save_compile_status(CompileRunStatusRecord(job_id=job.job_id, phase="starting"))
ready_capabilities = registry.configured_capability_ids()
```

Before running live SDK-backed compile jobs on a machine, install the pinned
bridge dependency once:

```bash
cd src/twinr/agent/self_coding/codex_driver/sdk_bridge
npm ci
```

The self-coding compile path defaults to `gpt-5-codex` with `high` reasoning
effort and a long compile budget. Override only when needed:

```bash
export TWINR_SELF_CODING_CODEX_MODEL=gpt-5-codex
export TWINR_SELF_CODING_CODEX_MODEL_REASONING_EFFORT=high
export TWINR_SELF_CODING_CODEX_TIMEOUT_SECONDS=900
```

Before the first live compile on a machine, run the explicit readiness preflight:

```bash
PYTHONPATH=src python3 -m twinr --env-file .env --self-coding-codex-self-test --self-coding-live-auth-check
```

For the minimum real product proof, run the full morning-briefing acceptance:

```bash
PYTHONPATH=src python3 -m twinr --env-file .env --self-coding-morning-briefing-acceptance
```

For operator-side capture validation of one already active learned skill, use
the web self-coding page or call the retest helper from Python.

## See also

- [component.yaml](./component.yaml)
- [AGENTS.md](./AGENTS.md)
- [modules/](./modules)
- [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)
- [REQUIREMENTS.md](./REQUIREMENTS.md)
